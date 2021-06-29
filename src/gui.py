# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Yeast ACDC GUI for correcting Segmentation and Tracking errors"""

import sys
import os
import re
import traceback
import time
from functools import partial

import cv2
import numpy as np
import pandas as pd
import scipy.optimize
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.draw
import skimage.exposure
import skimage.transform
from skimage import img_as_float
from skimage.color import gray2rgb

from PyQt5.QtCore import Qt, QFile, QTextStream, QSize
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
from myutils import download_model
from QtDarkMode import breeze_resources

# Interpret image data as row-major instead of col-major
pg.setConfigOption('imageAxisOrder', 'row-major') # best performance
np.random.seed(1568)

def qt_debug_trace():
    from PyQt5.QtCore import pyqtRemoveInputHook
    pyqtRemoveInputHook()
    import pdb; pdb.set_trace()

class Yeast_ACDC_GUI(QMainWindow):
    """Main Window."""

    def __init__(self, app, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.app = app
        self.num_screens = len(app.screens())

        # Center main window and determine location of slideshow window
        # depending on number of screens available
        mainWinWidth = 1366
        mainWinHeight = 768
        mainWinLeft = int(app.screens()[0].size().width()/2 - mainWinWidth/2)
        mainWinTop = int(app.screens()[0].size().height()/2 - mainWinHeight/2)

        if self.num_screens > 1:
            screen1 = app.screens()[0]
            screen2 = app.screens()[1]
            screen2Center = screen2.size().width()/2
            screen2Left = screen1.size().width()
            self.slideshowWinLeft = int(screen2Left+screen2Center-850/2)
            self.slideshowWinTop = int(screen1.size().height()/2 - 800/2)
        else:
            screen1 = app.screens()[0]
            self.slideshowWinLeft = int(screen1.size().width()-850)
            self.slideshowWinTop = int(screen1.size().height()/2 - 800/2)

        self.slideshowWin = None
        self.data_loaded = False
        self.setWindowTitle("Yeast ACDC - Segm&Track")
        self.setGeometry(mainWinLeft, mainWinTop, mainWinWidth, mainWinHeight)

        self.checkableButtons = []

        # Buttons added to QButtonGroup will be mutually exclusive
        self.checkableQButtonsGroup = QButtonGroup(self)
        self.checkableQButtonsGroup.setExclusive(False)

        self.gui_createActions()
        self.gui_createMenuBar()
        self.gui_createToolBars()

        self.gui_connectActions()
        self.gui_createStatusBar()

        self.gui_createGraphics()

        self.gui_createImg1Widgets()

        self.setEnabledToolbarButton(enabled=False)

        mainContainer = QtGui.QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QtGui.QGridLayout()
        mainLayout.addWidget(self.graphLayout, 0, 0, 1, 2)
        mainLayout.addLayout(self.img1_Widglayout, 1, 0)

        mainContainer.setLayout(mainLayout)

    def leaveEvent(self, event):
        if self.slideshowWin is not None:
            mainWinGeometry = self.frameGeometry()
            mainWinLeft = mainWinGeometry.left()
            mainWinTop = mainWinGeometry.top()
            mainWinWidth = mainWinGeometry.width()
            mainWinHeight = mainWinGeometry.height()
            mainWinRight = mainWinLeft+mainWinWidth
            mainWinBottom = mainWinTop+mainWinHeight

            slideshowWinGeometry = self.slideshowWin.frameGeometry()
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
                self.disableAutoActivateViewerWindow
            )

            if autoActivate:
                self.slideshowWin.setFocus(True)
                self.slideshowWin.activateWindow()

    def enterEvent(self, event):
        if self.slideshowWin is not None:
            mainWinGeometry = self.frameGeometry()
            mainWinLeft = mainWinGeometry.left()
            mainWinTop = mainWinGeometry.top()
            mainWinWidth = mainWinGeometry.width()
            mainWinHeight = mainWinGeometry.height()
            mainWinRight = mainWinLeft+mainWinWidth
            mainWinBottom = mainWinTop+mainWinHeight

            slideshowWinGeometry = self.slideshowWin.frameGeometry()
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
                self.disableAutoActivateViewerWindow
            )

            if autoActivate:
                self.setFocus(True)
                self.activateWindow()

    def gui_createImg1Widgets(self):
        self.img1_Widglayout = QtGui.QGridLayout()

        # Toggle contours/ID combobox
        row = 0
        self.drawIDsContComboBoxSegmItems = ['Draw IDs and contours',
                                             'Draw only IDs',
                                             'Draw only contours',
                                             'Draw nothing']
        self.drawIDsContComboBoxCcaItems = ['Draw only cell cycle info',
                                            'Draw cell cycle info and contours',
                                            'Draw IDs and contours',
                                            'Draw only IDs',
                                            'Draw only contours',
                                            'Draw nothing']
        self.drawIDsContComboBox = QComboBox()
        self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxSegmItems)
        # Always adjust combobox width to largest item
        self.drawIDsContComboBox.setSizeAdjustPolicy(
                    self.drawIDsContComboBox.AdjustToContents)

        self.img1_Widglayout.addWidget(
                self.drawIDsContComboBox, row, 1, 1, 10,
                alignment=Qt.AlignCenter)

        # z-slice scrollbar
        row += 1
        self.zSlice_scrollBar_img1 = QScrollBar(Qt.Horizontal)
        self.zSlice_scrollBar_img1.setFixedHeight(20)
        self.zSlice_scrollBar_img1.setDisabled(True)
        _z_label = QLabel('z-slice  ')
        _font = QtGui.QFont()
        _font.setPointSize(10)
        _z_label.setFont(_font)
        self.img1_Widglayout.addWidget(
                _z_label, row, 0, alignment=Qt.AlignRight)
        self.img1_Widglayout.addWidget(
                self.zSlice_scrollBar_img1, row, 1, 1, 10)

        # Fluorescent overlay alpha
        row += 2
        alphaScrollBar_label = QLabel('Overlay alpha  ')
        alphaScrollBar_label.setFont(_font)
        alphaScrollBar = QScrollBar(Qt.Horizontal)
        alphaScrollBar.setFixedHeight(20)
        alphaScrollBar.setMinimum(0)
        alphaScrollBar.setMaximum(20)
        alphaScrollBar.setValue(10)
        alphaScrollBar.setToolTip(
            'Control the alpha value of the overlay.\n'
            'alpha=0 results in NO overlay,\n'
            'alpha=1 results in only fluorescent data visible'
        )
        alphaScrollBar.setDisabled(True)
        self.alphaScrollBar = alphaScrollBar
        self.img1_Widglayout.addWidget(
            alphaScrollBar_label, row, 0
        )
        self.img1_Widglayout.addWidget(
            alphaScrollBar, row, 1, 1, 10
        )

        # Left, top, right, bottom
        self.img1_Widglayout.setContentsMargins(100, 0, 0, 0)


    def gui_createGraphics(self):
        self.graphLayout = pg.GraphicsLayoutWidget()

        # Left plot
        self.ax1 = pg.PlotItem()
        self.ax1.invertY(True)
        self.ax1.setAspectLocked(True)
        self.ax1.hideAxis('bottom')
        self.ax1.hideAxis('left')
        self.graphLayout.addItem(self.ax1, row=1, col=1)

        # Left image
        self.img1 = pg.ImageItem(np.zeros((512,512)))
        self.ax1.addItem(self.img1)

        # Left image histogram
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img1)
        self.graphLayout.addItem(hist, row=1, col=0)

        # Auto image adjustment button
        proxy = QtGui.QGraphicsProxyWidget()
        equalizeHistPushButton = QPushButton("Auto")
        equalizeHistPushButton.setStyleSheet(
               'QPushButton {background-color: #282828; color: #F0F0F0;}')
        proxy.setWidget(equalizeHistPushButton)
        self.graphLayout.addItem(proxy, row=0, col=0)
        self.equalizeHistPushButton = equalizeHistPushButton

        # Right plot
        self.ax2 = pg.PlotItem()
        self.ax2.setAspectLocked(True)
        self.ax2.invertY(True)
        self.ax2.hideAxis('bottom')
        self.ax2.hideAxis('left')
        self.graphLayout.addItem(self.ax2, row=1, col=2)


        # Right image
        self.img2 = pg.ImageItem(np.zeros((512,512)))
        self.ax2.addItem(self.img2)

        # Brush circle img1
        self.ax1_BrushCircle = pg.ScatterPlotItem()
        self.ax1_BrushCircle.setData([], [], symbol='o', pxMode=False,
                                 brush=pg.mkBrush((255,255,255,50)),
                                 pen=pg.mkPen(width=2))
        self.ax1.addItem(self.ax1_BrushCircle)

        # Eraser circle img2
        self.EraserCircle = pg.ScatterPlotItem()
        self.EraserCircle.setData([], [], symbol='o', pxMode=False,
                                 brush=None,
                                 pen=pg.mkPen(width=2, color='r'))
        self.ax2.addItem(self.EraserCircle)

        # Brush circle img2
        self.ax2_BrushCircle = pg.ScatterPlotItem()
        self.ax2_BrushCircle.setData([], [], symbol='o', pxMode=False,
                                 brush=pg.mkBrush((255,255,255,50)),
                                 pen=pg.mkPen(width=2))
        self.ax2.addItem(self.ax2_BrushCircle)


        # Experimental: brush cursors
        self.eraserCursor = QCursor(QIcon(":eraser.png").pixmap(30, 30))
        brushCursorPixmap = QIcon(":brush-cursor.png").pixmap(32, 32)
        self.brushCursor = QCursor(brushCursorPixmap, 16, 16)

        # Annotated metadata markers (ScatterPlotItem)
        self.binnedIDs_ScatterPlot = pg.ScatterPlotItem()
        self.binnedIDs_ScatterPlot.setData(
                                 [], [], symbol='t', pxMode=False,
                                 brush=pg.mkBrush((255,0,0,50)), size=15,
                                 pen=pg.mkPen(width=3, color='r'))
        self.ax2.addItem(self.binnedIDs_ScatterPlot)
        self.ripIDs_ScatterPlot = pg.ScatterPlotItem()
        self.ripIDs_ScatterPlot.setData(
                                 [], [], symbol='x', pxMode=False,
                                 brush=pg.mkBrush((255,0,0,50)), size=15,
                                 pen=pg.mkPen(width=2, color='r'))
        self.ax2.addItem(self.ripIDs_ScatterPlot)


        # Title
        self.titleLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        self.titleLabel.setText(
            'File --> Open or Open recent to start the process')
        self.graphLayout.addItem(self.titleLabel, row=0, col=1, colspan=2)

        # Current frame text
        self.frameLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        self.frameLabel.setText(' ')
        self.graphLayout.addItem(self.frameLabel, row=2, col=1, colspan=2)

    def gui_createGraphicsItems(self):
        maxID = self.data.segm_data.max()

        # Contour pens
        self.oldIDs_cpen = pg.mkPen(color=(200, 0, 0, 255*0.5), width=2)
        self.newIDs_cpen = pg.mkPen(color='r', width=3)

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

        # Create enough PlotDataItems and LabelItems to draw contours and IDs
        numItems = maxID+10
        self.ax1_ContoursCurves = []
        self.ax1_BudMothLines = []
        self.ax1_LabelItemsIDs = []
        self.ax2_LabelItemsIDs = []
        for i in range(numItems):
            # Contours on ax1
            ContCurve = pg.PlotDataItem()
            self.ax1_ContoursCurves.append(ContCurve)
            self.ax1.addItem(ContCurve)

            # Contours on ax1
            BudMothLine = pg.PlotDataItem()
            self.ax1_BudMothLines.append(BudMothLine)
            self.ax1.addItem(BudMothLine)

            # LabelItems on ax1
            ax1_IDlabel = pg.LabelItem()
            self.ax1_LabelItemsIDs.append(ax1_IDlabel)
            self.ax1.addItem(ax1_IDlabel)

            # LabelItems on ax2
            ax2_IDlabel = pg.LabelItem()
            self.ax2_LabelItemsIDs.append(ax2_IDlabel)
            self.ax2.addItem(ax2_IDlabel)



    def gui_connectGraphicsEvents(self):
        self.img1.hoverEvent = self.gui_hoverEventImg1
        self.img2.hoverEvent = self.gui_hoverEventImg2
        self.img1.mousePressEvent = self.gui_mousePressEventImg1
        self.img1.mouseMoveEvent = self.gui_mouseDragEventImg1
        self.img1.mouseReleaseEvent = self.gui_mouseReleaseEventImg1
        self.img2.mousePressEvent = self.gui_mousePressEventImg2
        self.img2.mouseMoveEvent = self.gui_mouseDragEventImg2
        self.img2.mouseReleaseEvent = self.gui_mouseReleaseEventImg2
        # self.graphLayout.mouseReleaseEvent = self.gui_mouseReleaseEvent


    def gui_mousePressEventImg2(self, event):
        mode = str(self.modeComboBox.currentText())
        left_click = event.button() == Qt.MouseButton.LeftButton
        mid_click = event.button() == Qt.MouseButton.MidButton
        right_click = event.button() == Qt.MouseButton.RightButton
        eraserON = self.eraserButton.isChecked()
        brushON = self.brushButton.isChecked()

        # Drag image if neither brush or eraser are On or Alt is pressed
        dragImg = (
            (left_click and not eraserON and not brushON) or
            (left_click and self.isAltDown)
        )

        # Enable dragging of the image window like pyqtgraph original code
        if dragImg:
            pg.ImageItem.mousePressEvent(self.img2, event)

        if mode == 'Viewer':
            return

        # Erase with brush and left click on the right image
        # NOTE: contours, IDs and rp will be updated
        # on gui_mouseReleaseEventImg2
        if left_click and self.eraserButton.isChecked() and not dragImg:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            Y, X = self.lab.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)
                self.yPressAx2, self.xPressAx2 = y, x
                self.isMouseDragImg2 = True
                # Keep a global mask to compute which IDs got erased
                self.erasedIDs = []
                brushSize = self.brushSizeSpinbox.value()
                mask = skimage.morphology.disk(brushSize, dtype=np.bool)
                ymin, xmin = ydata-brushSize, xdata-brushSize
                ymax, xmax = ydata+brushSize+1, xdata+brushSize+1
                self.erasedIDs.extend(self.lab[ymin:ymax, xmin:xmax][mask])
                self.lab[ymin:ymax, xmin:xmax][mask] = 0
                self.img2.updateImage()

        # Paint with brush and left click on the right image
        # NOTE: contours, IDs and rp will be updated
        # on gui_mouseReleaseEventImg2
        elif left_click and self.brushButton.isChecked() and not dragImg:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            Y, X = self.lab.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)
                self.yPressAx2, self.xPressAx2 = y, x

                self.isMouseDragImg2 = True

                brushSize = self.brushSizeSpinbox.value()
                mask = skimage.morphology.disk(brushSize, dtype=np.bool)
                ymin, xmin = ydata-brushSize, xdata-brushSize
                ymax, xmax = ydata+brushSize+1, xdata+brushSize+1
                maskedLab = self.lab[ymin:ymax, xmin:xmax][mask]
                self.ax2_brushID = [ID for ID in np.unique(maskedLab) if ID!=0]
                self.lab[ymin:ymax, xmin:xmax][mask] = self.ax2_brushID[0]
                self.img2.updateImage()

        # Delete entire ID (set to 0)
        elif mid_click and mode == 'Segmentation and Tracking':
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            delID = self.lab[ydata, xdata]
            if delID == 0:
                delID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here ID that you want to delete'
                )
                delID_prompt.exec_()
                if delID_prompt.cancel:
                    return
                else:
                    delID = delID_prompt.EntryID

            # Ask to propagate change to all future visited frames
            (UndoFutFrames, applyFutFrames, endFrame_i,
            doNotShowAgain) = self.propagateChange(
                                    delID, 'Delete cell ID',
                                    self.doNotShowAgain_DelID,
                                    self.UndoFutFrames_DelID,
                                    self.applyFutFrames_DelID)

            self.doNotShowAgain_DelID = doNotShowAgain
            self.UndoFutFrames_DelID = UndoFutFrames
            self.applyFutFrames_DelID = applyFutFrames

            self.current_frame_i = self.frame_i

            # Apply Edit ID to future frames if requested
            if applyFutFrames:
                self.app.setOverrideCursor(Qt.WaitCursor)
                # Store current data before going to future frames
                self.store_data()
                for i in range(self.frame_i+1, endFrame_i):
                    lab = self.allData_li[i]['labels']
                    if lab is None:
                        break

                    lab[lab==delID] = 0

                    # Store change
                    self.allData_li[i]['labels'] = lab.copy()
                    # Get the rest of the stored metadata based on the new lab
                    self.frame_i = i
                    self.get_data()
                    self.update_rp_metadata(draw=False)
                    self.store_data()

                self.app.restoreOverrideCursor()

            # Back to current frame
            if applyFutFrames:
                self.frame_i = self.current_frame_i
                self.get_data()
                self.update_rp_metadata(draw=False)


            # Store undo state before modifying stuff
            self.storeUndoRedoStates(UndoFutFrames)
            self.lab[self.lab==delID] = 0

            # Update data (rp, etc)
            prev_IDs = [obj.label for obj in self.rp]
            self.update_rp()

            self.img2.setImage(self.lab)

            # Remove contour and LabelItem of deleted ID
            self.ax1_ContoursCurves[delID-1].setData([], [])
            self.ax1_LabelItemsIDs[delID-1].setText('')
            self.ax2_LabelItemsIDs[delID-1].setText('')

            self.checkIDs_LostNew()

        # Separate bud
        elif (right_click or left_click) and self.separateBudButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            ID = self.lab[ydata, xdata]
            if ID == 0:
                sepID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here ID that you want to split'
                )
                sepID_prompt.exec_()
                if sepID_prompt.cancel:
                    return
                else:
                    ID = sepID_prompt.EntryID



            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            max_ID = self.lab.max()

            if right_click:
                self.lab, success = self.auto_separate_bud_ID(
                                             ID, self.lab, self.rp,
                                             max_ID, enforce=True)
            else:
                success = False

            # If automatic bud separation was not successfull call manual one
            if not success:
                self.disableAutoActivateViewerWindow = True
                paint_out = apps.my_paint_app(
                                self.lab, ID, self.rp, del_small_obj=True,
                                overlay_img=self.img1.image)
                if paint_out.cancel:
                    self.disableAutoActivateViewerWindow = False
                    self.separateBudButton.setChecked(False)
                    return
                paint_out_lab = paint_out.sep_bud_label
                self.lab[paint_out_lab!=0] = paint_out_lab[paint_out_lab!=0]
                self.lab[paint_out.small_obj_mask] = 0
                # Apply eraser mask only to clicked ID
                eraser_mask = np.logical_and(paint_out.eraser_mask, self.lab==ID)
                self.lab[eraser_mask] = 0
                for yy, xx in paint_out.coords_delete:
                    del_ID = self.lab[yy, xx]
                    self.lab[self.lab == del_ID] = 0

                self.disableAutoActivateViewerWindow = False

            # Update data (rp, etc)
            prev_IDs = [obj.label for obj in self.rp]
            self.update_rp()

            # Repeat tracking
            self.tracking()

            # Update all images
            self.updateALLimg()

            # Uncheck separate bud button
            self.separateBudButton.setChecked(False)

        # Merge IDs
        elif right_click and self.mergeIDsButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            ID = self.lab[ydata, xdata]
            if ID == 0:
                mergeID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here first ID that you want to merge'
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
            xdata, ydata = int(round(x)), int(round(y))
            ID = self.lab[ydata, xdata]
            if ID == 0:
                editID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here ID that you want to replace with a new one'
                )
                editID_prompt.exec_()
                if editID_prompt.cancel:
                    return
                else:
                    ID = editID_prompt.EntryID
                    obj_idx = self.IDs.index(ID)
                    y, x = self.rp[obj_idx].centroid
                    xdata, ydata = int(round(x)), int(round(y))

            self.disableAutoActivateViewerWindow = True
            prev_IDs = [obj.label for obj in self.rp]
            editID = apps.editID_QWidget(ID, prev_IDs)
            editID.exec_()
            if editID.cancel:
                self.disableAutoActivateViewerWindow = False
                self.editID_Button.setChecked(False)
                return

            # Ask to propagate change to all future visited frames
            (UndoFutFrames, applyFutFrames, endFrame_i,
            doNotShowAgain) = self.propagateChange(
                                    ID, 'Edit ID',
                                    self.doNotShowAgain_EditID,
                                    self.UndoFutFrames_EditID,
                                    self.applyFutFrames_EditID,
                                    applyTrackingB=True)

            if UndoFutFrames is None:
                return

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(UndoFutFrames)

            for old_ID, new_ID in editID.how:
                if new_ID in prev_IDs:
                    tempID = self.lab.max() + 1
                    self.lab[self.lab == old_ID] = tempID
                    self.lab[self.lab == new_ID] = old_ID
                    self.lab[self.lab == tempID] = new_ID

                    # Clear labels IDs of the swapped IDs
                    self.ax2_LabelItemsIDs[old_ID-1].setText('')
                    self.ax1_LabelItemsIDs[old_ID-1].setText('')
                    self.ax2_LabelItemsIDs[new_ID-1].setText('')
                    self.ax1_LabelItemsIDs[new_ID-1].setText('')


                    old_ID_idx = prev_IDs.index(old_ID)
                    new_ID_idx = prev_IDs.index(new_ID)
                    self.rp[old_ID_idx].label = new_ID
                    self.rp[new_ID_idx].label = old_ID
                    self.drawID_and_Contour(
                        self.rp[old_ID_idx],
                        drawContours=False
                    )
                    self.drawID_and_Contour(
                        self.rp[new_ID_idx],
                        drawContours=False
                    )

                    # Append information for replicating the edit in tracking
                    # List of tuples (y, x, replacing ID)
                    obj = self.rp[old_ID_idx]
                    y, x = obj.centroid
                    y, x = int(round(y)), int(round(x))
                    self.editID_info.append((y, x, new_ID))
                    obj = self.rp[new_ID_idx]
                    y, x = obj.centroid
                    y, x = int(round(y)), int(round(x))
                    self.editID_info.append((y, x, old_ID))
                else:
                    self.lab[self.lab == old_ID] = new_ID
                    # Clear labels IDs of the swapped IDs
                    self.ax2_LabelItemsIDs[old_ID-1].setText('')
                    self.ax1_LabelItemsIDs[old_ID-1].setText('')

                    old_ID_idx = prev_IDs.index(old_ID)
                    self.rp[old_ID_idx].label = new_ID
                    self.drawID_and_Contour(
                        self.rp[old_ID_idx],
                        drawContours=False
                    )
                    # Append information for replicating the edit in tracking
                    # List of tuples (y, x, replacing ID)
                    obj = self.rp[old_ID_idx]
                    y, x = obj.centroid
                    y, x = int(round(y)), int(round(x))
                    self.editID_info.append((y, x, new_ID))

            # Update rps
            self.update_rp()

            # Since we manually changed an ID we don't want to repeat tracking
            self.checkIDs_LostNew()

            # Update colors for the edited IDs
            self.updateLookuptable()

            self.img2.setImage(self.lab)
            self.editID_Button.setChecked(False)

            self.disableAutoActivateViewerWindow = True


            # Perform desired action on future frames
            self.doNotShowAgain_EditID = doNotShowAgain
            self.UndoFutFrames_EditID = UndoFutFrames
            self.applyFutFrames_EditID = applyFutFrames

            self.current_frame_i = self.frame_i

            if applyFutFrames:
                self.app.setOverrideCursor(Qt.WaitCursor)
                # Store data for current frame
                self.store_data()
                for i in range(self.frame_i+1, endFrame_i):
                    self.frame_i = i
                    self.get_data()
                    self.update_rp_metadata(draw=False)
                    if self.onlyTracking:
                        self.tracking(enforce=True)
                    else:
                        for old_ID, new_ID in editID.how:
                            if new_ID in prev_IDs:
                                tempID = self.lab.max() + 1
                                self.lab[self.lab == old_ID] = tempID
                                self.lab[self.lab == new_ID] = old_ID
                                self.lab[self.lab == tempID] = new_ID
                            else:
                                self.lab[self.lab == old_ID] = new_ID
                        self.update_rp(draw=False)
                    self.store_data()

                # Back to current frame
                self.frame_i = self.current_frame_i
                self.get_data()
                self.update_rp_metadata(draw=False)
                self.app.restoreOverrideCursor()


        # Annotate cell as removed from the analysis
        elif right_click and self.binCellButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            ID = self.lab[ydata, xdata]
            if ID == 0:
                binID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to remove from the analysis'
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
                                    self.doNotShowAgain_BinID,
                                    self.UndoFutFrames_BinID,
                                    self.applyFutFrames_BinID)

            self.doNotShowAgain_BinID = doNotShowAgain
            self.UndoFutFrames_BinID = UndoFutFrames
            self.applyFutFrames_BinID = applyFutFrames

            self.current_frame_i = self.frame_i

            # Apply Edit ID to future frames if requested
            if applyFutFrames:
                self.app.setOverrideCursor(Qt.WaitCursor)
                # Store current data before going to future frames
                self.store_data()
                for i in range(self.frame_i+1, endFrame_i):
                    self.frame_i = i
                    self.get_data()
                    if ID in self.binnedIDs:
                        self.binnedIDs.remove(ID)
                    else:
                        self.binnedIDs.add(ID)
                    self.update_rp_metadata(draw=False)
                    self.store_data()

                self.app.restoreOverrideCursor()

            # Back to current frame
            if applyFutFrames:
                self.frame_i = self.current_frame_i
                self.get_data()
                self.update_rp_metadata(draw=False)

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(UndoFutFrames)

            if ID in self.binnedIDs:
                self.binnedIDs.remove(ID)
            else:
                self.binnedIDs.add(ID)

            self.update_rp_metadata()

            # Gray out ore restore binned ID
            self.updateLookuptable()

            self.binCellButton.setChecked(False)

        # Annotate cell as dead
        elif right_click and self.ripCellButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            ID = self.lab[ydata, xdata]
            if ID == 0:
                ripID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as dead'
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
                                    self.doNotShowAgain_RipID,
                                    self.UndoFutFrames_RipID,
                                    self.applyFutFrames_RipID)

            self.doNotShowAgain_RipID = doNotShowAgain
            self.UndoFutFrames_RipID = UndoFutFrames
            self.applyFutFrames_RipID = applyFutFrames

            self.current_frame_i = self.frame_i

            # Apply Edit ID to future frames if requested
            if applyFutFrames:
                self.app.setOverrideCursor(Qt.WaitCursor)
                # Store current data before going to future frames
                self.store_data()
                for i in range(self.frame_i+1, endFrame_i):
                    self.frame_i = i
                    self.get_data()
                    if ID in self.ripIDs:
                        self.ripIDs.remove(ID)
                    else:
                        self.ripIDs.add(ID)
                    self.update_rp_metadata(draw=False)
                    self.store_data()
                self.app.restoreOverrideCursor()

            # Back to current frame
            if applyFutFrames:
                self.frame_i = self.current_frame_i
                self.get_data()
                self.update_rp_metadata(draw=False)

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(UndoFutFrames)

            if ID in self.ripIDs:
                self.ripIDs.remove(ID)
            else:
                self.ripIDs.add(ID)

            self.update_rp_metadata()

            # Gray out dead ID
            self.updateLookuptable()

            self.ripCellButton.setChecked(False)

    def getPolygonBrush(self, yxc2):
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
                                                    [x3, x4, x6, x5])
        else:
            rr_poly, cc_poly = [], []

        self.yPressAx2, self.xPressAx2 = y2, x2
        return rr_poly, cc_poly




    def gui_mouseDragEventImg2(self, event):
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

        # Eraser dragging mouse --> keep erasing
        if self.isMouseDragImg2 and self.eraserButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            brushSize = self.brushSizeSpinbox.value()
            mask = skimage.morphology.disk(brushSize, dtype=np.bool)
            rrPoly, ccPoly = self.getPolygonBrush((y, x))
            ymin, xmin = ydata-brushSize, xdata-brushSize
            ymax, xmax = ydata+brushSize+1, xdata+brushSize+1
            self.erasedIDs.extend(self.lab[ymin:ymax, xmin:xmax][mask])
            self.erasedIDs.extend(self.lab[rrPoly, ccPoly])
            self.lab[ymin:ymax, xmin:xmax][mask] = 0
            self.lab[rrPoly, ccPoly] = 0
            self.img2.updateImage()

        # Brush paint dragging mouse --> keep painting
        if self.isMouseDragImg2 and self.brushButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            brushSize = self.brushSizeSpinbox.value()
            mask = skimage.morphology.disk(brushSize, dtype=np.bool)
            rrPoly, ccPoly = self.getPolygonBrush((y, x))
            ymin, xmin = ydata-brushSize, xdata-brushSize
            ymax, xmax = ydata+brushSize+1, xdata+brushSize+1
            self.lab[ymin:ymax, xmin:xmax][mask] = self.ax2_brushID
            self.lab[rrPoly, ccPoly] = self.ax2_brushID
            self.img2.updateImage()

    def gui_mouseReleaseEventImg2(self, event):
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return


        # Eraser mouse release --> update IDs and contours
        if self.isMouseDragImg2 and self.eraserButton.isChecked():
            self.isMouseDragImg2 = False
            erasedIDs = np.unique(self.erasedIDs)

            self.img2.setImage(self.lab)

            # Update data (rp, etc)
            prev_IDs = [obj.label for obj in self.rp]
            self.update_rp()

            self.update_IDsContours(
                prev_IDs, newIDs=erasedIDs
            )

        # Brush mouse release --> update IDs and contours
        elif self.isMouseDragImg2 and self.brushButton.isChecked():
            self.isMouseDragImg2 = False

            self.img2.setImage(self.lab)

            # Update data (rp, etc)
            prev_IDs = [obj.label for obj in self.rp]
            self.update_rp()

            self.update_IDsContours(
                prev_IDs, newIDs=self.ax2_brushID
            )

        # Merge IDs
        elif self.mergeIDsButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            ID = self.lab[ydata, xdata]
            if ID == 0:
                mergeID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to merge with ID '
                         f'{self.firstID}'
                )
                mergeID_prompt.exec_()
                if mergeID_prompt.cancel:
                    return
                else:
                    ID = mergeID_prompt.EntryID

            self.lab[self.lab==ID] = self.firstID

            # Mask to keep track of which ID needs redrawing of the contours
            mergedID_mask = self.lab==self.firstID

            # Update data (rp, etc)
            prev_IDs = [obj.label for obj in self.rp]
            self.update_rp()

            # Repeat tracking
            self.tracking()

            newID = self.lab[mergedID_mask][0]
            self.img2.updateImage()
            self.update_IDsContours(
                prev_IDs, newIDs=[newID]
            )
            self.mergeIDsButton.setChecked(False)


    def gui_mouseReleaseEventImg1(self, event):
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

        if self.mergeIDsButton.isChecked():
            # Allow right-click actions on both images
            self.gui_mouseReleaseEventImg2(event)

        # Assign mother to bud
        elif self.assignMothBudButton.isChecked() and self.clickedOnBud:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            ID = self.lab[ydata, xdata]
            if ID == 0:
                mothID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as mother cell'
                )
                mothID_prompt.exec_()
                if mothID_prompt.cancel:
                    return
                else:
                    ID = mothID_prompt.EntryID
                    obj_idx = self.IDs.index(ID)
                    y, x = self.rp[obj_idx].centroid
                    xdata, ydata = int(round(x)), int(round(y))

            relationship = self.cca_df.at[ID, 'relationship']
            ccs = self.cca_df.at[ID, 'cell_cycle_stage']

            if relationship == 'bud' and self.frame_i > 0:
                txt = (f'You clicked on ID {ID} which is a BUD.\n'
                       'To assign a bud to a cell start by clicking on a bud '
                       'and release on a cell in G1')
                msg = QtGui.QMessageBox()
                msg.critical(
                    self, 'Released on a bud', txt, msg.Ok
                )
                return

            elif ccs != 'G1' and self.frame_i > 0:
                txt = (f'You clicked on a cell (ID={ID}) which is NOT in G1.\n'
                       'To assign a bud to a cell start by clicking on a bud '
                       'and release on a cell in G1')
                msg = QtGui.QMessageBox()
                msg.critical(
                    self, 'Released on a cell NOT in G1', txt, msg.Ok
                )
                return

            self.xClickMoth, self.yClickMoth = xdata, ydata
            self.assignMothBud()
            self.assignMothBudButton.setChecked(False)
            self.draw_MothBudTempLine = False
            self.BudMothTempLine.setData([], [])

    def gui_mousePressEventImg1(self, event):
        mode = str(self.modeComboBox.currentText())
        is_cca_on = mode == 'Cell cycle analysis'
        right_click = event.button() == Qt.MouseButton.RightButton
        left_click = event.button() == Qt.MouseButton.LeftButton
        mid_click = event.button() == Qt.MouseButton.MidButton

        dragImg = (
            (left_click and not self.brushButton.isChecked()) or
            (left_click and self.isAltDown)
        )

        # Enable dragging of the image window like pyqtgraph original code
        if dragImg:
            pg.ImageItem.mousePressEvent(self.img1, event)

        if mode == 'Viewer':
            return

        # Paint new IDs with brush and left click on the left image
        if left_click and self.brushButton.isChecked() and not dragImg:
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            Y, X = self.lab.shape
            brushSize = self.brushSizeSpinbox.value()
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                # Update brush ID. Take care of disappearing cells to remember
                # to not use their IDs anymore in the future
                if self.lab.max()+1 > self.brushID:
                    self.brushID = self.lab.max()+1
                else:
                    self.brushID += 1

                mask = skimage.morphology.disk(brushSize, dtype=np.bool)
                ymin, xmin = ydata-brushSize, xdata-brushSize
                ymax, xmax = ydata+brushSize+1, xdata+brushSize+1

                # Draw new objects below existing ones
                localLab = self.lab[ymin:ymax, xmin:xmax]
                mask[localLab!=0] = False
                self.lab[ymin:ymax, xmin:xmax][mask] = self.brushID

                # Update data (rp, etc)
                prev_IDs = [obj.label for obj in self.rp]
                self.update_rp()

                # Repeat tracking
                self.tracking(enforce=True)
                newIDs = [self.lab[ymin:ymax, xmin:xmax][mask][0]]

                # Update colors to include a new color for the new ID
                self.img2.setImage(self.lab)
                self.updateLookuptable()

                # Update contours
                self.update_IDsContours(prev_IDs, newIDs=newIDs)


        # Allow right-click actions on both images
        elif right_click and mode == 'Segmentation and Tracking':
            self.gui_mousePressEventImg2(event)

        # Annotate cell cycle division
        elif right_click and is_cca_on and not self.assignMothBudButton.isChecked():
            if self.frame_i <= 0:
                return

            if self.cca_df is None:
                return

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            ID = self.lab[ydata, xdata]
            if ID == 0:
                divID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as divided'
                )
                divID_prompt.exec_()
                if divID_prompt.cancel:
                    return
                else:
                    ID = divID_prompt.EntryID
                    obj_idx = self.IDs.index(ID)
                    y, x = self.rp[obj_idx].centroid
                    xdata, ydata = int(round(x)), int(round(y))

            # Annotate or undo division
            self.manualCellCycleAnnotation(ID)

        # Assign bud to mother (mouse down on bud)
        elif right_click and self.assignMothBudButton.isChecked():
            self.clickedOnBud = False
            if self.cca_df is None:
                return

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            ID = self.lab[ydata, xdata]
            if ID == 0:
                budID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID of a bud you want to correct mother assignment'
                )
                budID_prompt.exec_()
                if budID_prompt.cancel:
                    return
                else:
                    ID = budID_prompt.EntryID

            obj_idx = self.IDs.index(ID)
            y, x = self.rp[obj_idx].centroid
            xdata, ydata = int(round(x)), int(round(y))

            relationship = self.cca_df.at[ID, 'relationship']
            if relationship != 'bud' and self.frame_i > 0:
                txt = (f'You clicked on ID {ID} which is NOT a bud.\n'
                       'To assign a bud to a cell start by clicking on a bud '
                       'and release on a cell in G1')
                msg = QtGui.QMessageBox()
                msg.critical(
                    self, 'Not a bud', txt, msg.Ok
                )
                return
            self.clickedOnBud = True
            self.xClickBud, self.yClickBud = xdata, ydata
            self.draw_MothBudTempLine = True

        # Allow mid-click actions on both images
        elif mid_click and mode == 'Segmentation and Tracking':
            self.gui_mousePressEventImg2(event)

    def annotateDivision(self, cca_df, ID, relID, ccs, ccs_relID):
        # Correct as follows:
        # If S then correct to G1 and +1 on generation number
        store = False
        if ccs == 'S':
            cca_df.at[ID, 'cell_cycle_stage'] = 'G1'
            gen_num_clickedID = cca_df.at[ID, 'generation_num']
            cca_df.at[ID, 'generation_num'] += 1
            cca_df.at[ID, 'division_frame_i'] = self.frame_i
            cca_df.at[relID, 'cell_cycle_stage'] = 'G1'
            gen_num_relID = cca_df.at[relID, 'generation_num']
            cca_df.at[relID, 'generation_num'] += 1
            cca_df.at[relID, 'division_frame_i'] = self.frame_i
            if gen_num_clickedID < gen_num_relID:
                cca_df.at[ID, 'relationship'] = 'mother'
            else:
                cca_df.at[relID, 'relationship'] = 'mother'
            store = True
        return store

    def undoDivisionAnnotation(self, cca_df, ID, relID, ccs, ccs_relID):
        # Correct as follows:
        # If G1 then correct to S and -1 on generation number
        store = False
        if ccs == 'G1':
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
        # Correct current frame
        ccs = self.cca_df.at[ID, 'cell_cycle_stage']
        relID = self.cca_df.at[ID, 'relative_ID']
        ccs_relID = self.cca_df.at[relID, 'cell_cycle_stage']
        if ccs == 'S':
            store = self.annotateDivision(
                                    self.cca_df, ID, relID, ccs, ccs_relID)
            if store:
                self.store_cca_df()
        else:
            store = self.undoDivisionAnnotation(
                                    self.cca_df, ID, relID, ccs, ccs_relID)
            if store:
                self.store_cca_df()

        obj_idx = self.IDs.index(ID)
        relObj_idx = self.IDs.index(relID)
        rp_ID = self.rp[obj_idx]
        rp_relID = self.rp[relObj_idx]

        # Update cell cycle info LabelItems
        self.drawID_and_Contour(rp_ID, drawContours=False)
        self.drawID_and_Contour(rp_relID, drawContours=False)


        # Correct future frames
        for i in range(self.frame_i+1, self.num_segm_frames):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break
            else:
                ccs = cca_df_i.at[ID, 'cell_cycle_stage']
                relID = cca_df_i.at[ID, 'relative_ID']
                ccs_relID = cca_df_i.at[relID, 'cell_cycle_stage']
                if ccs == 'S':
                    store = self.annotateDivision(
                                        cca_df_i, ID, relID, ccs, ccs_relID)
                    if store:
                        self.store_cca_df(frame_i=i, cca_df=cca_df_i)
                else:
                    store = self.undoDivisionAnnotation(
                                        cca_df_i, ID, relID, ccs, ccs_relID)
                    if store:
                        self.store_cca_df(frame_i=i, cca_df=cca_df_i)

        # Correct past frames
        for i in range(self.frame_i-1, 0, -1):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            ccs = cca_df_i.at[ID, 'cell_cycle_stage']
            relID = cca_df_i.at[ID, 'relative_ID']
            ccs_relID = cca_df_i.at[relID, 'cell_cycle_stage']
            if ccs == 'S':
                # We correct only those frames in which the ID was in 'S'
                break
            else:
                store = self.undoDivisionAnnotation(
                                       cca_df_i, ID, relID, ccs, ccs_relID)
                if store:
                    self.store_cca_df(frame_i=i, cca_df=cca_df_i)

    def checkMothEligibility(self, budID, new_mothID):
        """Check the new mother is in G1 for the entire life of the bud"""

        eligible = True

        # Check future frames
        for i in range(self.frame_i, self.num_segm_frames):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                return eligible

            is_still_bud = cca_df_i.at[budID, 'relationship'] == 'bud'
            if not is_still_bud:
                # Bud life ended
                return eligible
            ccs = cca_df_i.at[new_mothID, 'cell_cycle_stage']
            if ccs != 'G1':
                err_msg = (
                    f'The requested cell in G1 (ID={new_mothID}) at future frame {i+1} '
                    'has a bud assigned to it, therefore it cannot be '
                    f'assigned as the mother of bud ID {budID}.\n'
                    f'You can assign a cell as the mother of bud ID {budID} '
                    'only if this cell is in G1 for the entire life of the bud.\n\n'
                    'One possible solution is to click on "cancel", go to '
                    f'frame {i+1} and  assign the bud of cell {new_mothID} '
                    'to another cell.\n'
                    f'A second solution is to assign bud ID {budID} to cell '
                    f'{new_mothID} anyway by clicking "Yes". However to ensure correctness of '
                    'future assignments the system will delete any cell cycle '
                    f'information from frame {i+1} to the end. Therefore, you '
                    'will have to visit those frames again.\n\n'
                    'The deletion of cell cycle information CANNOT BE UNDONE! '
                    'However, if you do not save no cell cycle information '
                    'saved on the hard drive will be removed.\n\n'
                    'Apply assignment or cancel process?')
                msg = QtGui.QMessageBox()
                enforce_assignment = msg.warning(
                   self, 'Cell not eligible', err_msg, msg.Apply | msg.Cancel
                )
                if enforce_assignment == msg.Cancel:
                    eligible = False
                else:
                    self.del_future_cca_df(i)
                return eligible

        # Check past frames
        for i in range(self.frame_i-1, 0, -1):
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)

            is_bud_existing = budID in cca_df_i.index
            if not is_bud_existing:
                # Bud was not emerged yet
                return eligible

            ccs = cca_df_i.at[new_mothID, 'cell_cycle_stage']
            if ccs != 'G1':
                err_msg = (
                    f'The requested cell in G1 (ID={new_mothID}) at past frame {i+1} '
                    'has a bud assigned to it, therefore it cannot be '
                    f'assigned as mother of bud ID {budID}.\n'
                    f'You can assign a cell as the mother of bud ID {budID} '
                    'only if this cell is in G1 for the entire life of the bud.\n'
                    f'One possible solution is to first go to frame {i+1} and '
                    f'assign the bud of cell {new_mothID} to another cell.')
                msg = QtGui.QMessageBox()
                msg.critical(
                   self, 'Cell not eligible', err_msg, msg.Ok
                )
                eligible = False
                return eligible

    def getStatusMothBeforeBudding(self, budID, curr_mothID):
        # Get status of the current mother before it had budID assigned to it
        for i in range(self.frame_i-1, 0, -1):
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)

            is_bud_existing = budID in cca_df_i.index
            if not is_bud_existing:
                # Bud was not emerged yet
                return cca_df_i.loc[curr_mothID]


    def assignMothBud(self):
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
        budID = self.lab[self.yClickBud, self.xClickBud]
        new_mothID = self.lab[self.yClickMoth, self.xClickMoth]

        if budID == new_mothID:
            return

        # Allow partial initialization of cca_df with mouse
        if self.frame_i == 0 and budID!=new_mothID:
            self.cca_df.at[budID, 'relationship'] = 'bud'
            self.cca_df.at[budID, 'generation_num'] = 0
            self.cca_df.at[budID, 'relative_ID'] = new_mothID
            self.cca_df.at[budID, 'cell_cycle_stage'] = 'S'
            self.cca_df.at[new_mothID, 'relative_ID'] = budID
            self.cca_df.at[new_mothID, 'generation_num'] = 2
            self.cca_df.at[new_mothID, 'cell_cycle_stage'] = 'S'
            bud_obj_idx = self.IDs.index(budID)
            new_moth_obj_idx = self.IDs.index(new_mothID)
            rp_budID = self.rp[bud_obj_idx]
            rp_new_mothID = self.rp[new_moth_obj_idx]
            self.drawID_and_Contour(rp_budID, drawContours=False)
            self.drawID_and_Contour(rp_new_mothID, drawContours=False)
            return

        curr_mothID = self.cca_df.at[budID, 'relative_ID']

        eligible = self.checkMothEligibility(budID, new_mothID)
        if not eligible:
            return

        curr_moth_cca = self.getStatusMothBeforeBudding(budID, curr_mothID)

        # Correct current frames and update LabelItems
        self.cca_df.at[budID, 'relative_ID'] = new_mothID

        self.cca_df.at[new_mothID, 'relative_ID'] = budID
        self.cca_df.at[new_mothID, 'cell_cycle_stage'] = 'S'

        self.cca_df.loc[curr_mothID] = curr_moth_cca

        bud_obj_idx = self.IDs.index(budID)
        new_moth_obj_idx = self.IDs.index(new_mothID)
        curr_moth_obj_idx = self.IDs.index(curr_mothID)
        rp_budID = self.rp[bud_obj_idx]
        rp_new_mothID = self.rp[new_moth_obj_idx]
        rp_curr_mothID = self.rp[curr_moth_obj_idx]

        self.drawID_and_Contour(rp_budID, drawContours=False)
        self.drawID_and_Contour(rp_new_mothID, drawContours=False)
        self.drawID_and_Contour(rp_curr_mothID, drawContours=False)

        # Correct future frames
        for i in range(self.frame_i+1, self.num_segm_frames):
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)

            if cca_df_i is None:
                # ith frame was not visited yet
                break

            cca_df_i.at[budID, 'relative_ID'] = new_mothID

            cca_df_i.at[new_mothID, 'relative_ID'] = budID
            cca_df_i.at[new_mothID, 'cell_cycle_stage'] = 'S'

            cca_df_i.loc[curr_mothID] = curr_moth_cca

            self.store_cca_df(frame_i=i, cca_df=cca_df_i)



        # Correct past frames
        for i in range(self.frame_i-1, 0, -1):
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)

            is_bud_existing = budID in cca_df_i.index
            if not is_bud_existing:
                # Bud was not emerged yet
                break

            cca_df_i.at[budID, 'relative_ID'] = new_mothID

            cca_df_i.at[new_mothID, 'relative_ID'] = budID
            cca_df_i.at[new_mothID, 'cell_cycle_stage'] = 'S'

            cca_df_i.loc[curr_mothID] = curr_moth_cca

            self.store_cca_df(frame_i=i, cca_df=cca_df_i)


    def gui_mouseDragEventImg1(self, event):
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

    def gui_hoverEventImg1(self, event):
        self.hoverEventImg1 = event
        # Update x, y, value label bottom right
        try:
            x, y = event.pos()
            xdata, ydata = int(round(x)), int(round(y))
            _img = self.img1.image
            Y, X = _img.shape[:2]
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                val = _img[ydata, xdata]
                maxVal = _img.max()
                ID = self.lab[ydata, xdata]
                maxID = self.lab.max()
                try:
                    self.wcLabel.setText(
                        f'(x={x:.2f}, y={y:.2f}, value={val:.2f}, '
                        f'max={maxVal:.2f}, ID={ID}, max_ID={maxID})'
                    )
                except:
                    val = [v for v in val]
                    self.wcLabel.setText(
                            f'(x={x:.2f}, y={y:.2f}, value={val})'
                    )
            else:
                self.wcLabel.setText(f'')
        except:
            self.wcLabel.setText(f'')

        # Draw Brush circle
        drawCircle = self.brushButton.isChecked() and not event.isExit()
        try:
            if drawCircle:
                x, y = event.pos()
                xdata, ydata = int(round(x)), int(round(y))
                _img = self.img2.image
                Y, X = _img.shape
                if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                    size = self.brushSizeSpinbox.value()*2
                    self.ax1_BrushCircle.setData([x], [y],
                                                   size=size)
            else:
                self.ax1_BrushCircle.setData([], [])
        except:
            # traceback.print_exc()
            self.ax1_BrushCircle.setData([], [])

        if self.draw_MothBudTempLine:
            try:
                x, y = event.pos()
                y2, x2 = y, x
                xdata, ydata = int(round(x)), int(round(y))
                y1, x1 = self.yClickBud, self.xClickBud
                ID = self.lab[ydata, xdata]
                if ID == 0:
                    self.BudMothTempLine.setData([x1, x2], [y1, y2])
                else:
                    obj_idx = self.IDs.index(ID)
                    obj = self.rp[obj_idx]
                    y2, x2 = obj.centroid
                    self.BudMothTempLine.setData([x1, x2], [y1, y2])
            except:
                # traceback.print_exc()
                self.BudMothTempLine.setData([], [])

    def gui_hoverEventImg2(self, event):
        self.hoverEventImg2 = event
        # Update x, y, value label bottom right
        try:
            x, y = event.pos()
            xdata, ydata = int(round(x)), int(round(y))
            _img = self.img2.image
            Y, X = _img.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                val = _img[ydata, xdata]
                self.wcLabel.setText(
                    f'(x={x:.2f}, y={y:.2f}, value={val:.0f}, max={_img.max()})'
                )
            else:
                self.wcLabel.setText(f'')
        except:
            # traceback.print_exc()
            self.wcLabel.setText(f'')

        # Draw eraser circle
        drawCircle = self.eraserButton.isChecked() and not event.isExit()
        try:
            if drawCircle:
                x, y = event.pos()
                xdata, ydata = int(round(x)), int(round(y))
                _img = self.img2.image
                Y, X = _img.shape
                if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                    size = self.brushSizeSpinbox.value()*2
                    self.EraserCircle.setData([x], [y], size=size)
            else:
                self.EraserCircle.setData([], [])
        except:
            self.EraserCircle.setData([], [])

        # Draw Brush circle
        drawCircle = self.brushButton.isChecked() and not event.isExit()
        try:
            if drawCircle:
                x, y = event.pos()
                xdata, ydata = int(round(x)), int(round(y))
                _img = self.img2.image
                Y, X = _img.shape
                if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                    size = self.brushSizeSpinbox.value()*2
                    self.ax2_BrushCircle.setData([x], [y],
                                                   size=size)
            else:
                self.ax2_BrushCircle.setData([], [])
        except:
            # traceback.print_exc()
            self.ax2_BrushCircle.setData([], [])


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
        toolbarSize = 34

        # File toolbar
        fileToolBar = self.addToolBar("File")
        fileToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        fileToolBar.setMovable(False)
        # fileToolBar.addAction(self.newAction)
        fileToolBar.addAction(self.openAction)
        fileToolBar.addAction(self.saveAction)
        fileToolBar.addAction(self.showInExplorerAction)
        fileToolBar.addAction(self.reloadAction)
        fileToolBar.addAction(self.undoAction)
        fileToolBar.addAction(self.redoAction)

        self.undoAction.setEnabled(False)
        self.redoAction.setEnabled(False)

        # Navigation toolbar
        navigateToolBar = QToolBar("Navigation", self)
        navigateToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(navigateToolBar)
        navigateToolBar.addAction(self.prevAction)
        navigateToolBar.addAction(self.nextAction)

        self.slideshowButton = QToolButton(self)
        self.slideshowButton.setIcon(QIcon(":eye-plus.svg"))
        self.slideshowButton.setCheckable(True)
        self.slideshowButton.setShortcut('Ctrl+W')
        self.slideshowButton.setToolTip('Open slideshow (Ctrl+W)')
        navigateToolBar.addWidget(self.slideshowButton)

        self.overlayButton = QToolButton(self)
        self.overlayButton.setIcon(QIcon(":overlay.svg"))
        self.overlayButton.setCheckable(True)
        self.overlayButton.setToolTip('Overlay fluorescent image')
        navigateToolBar.addWidget(self.overlayButton)
        self.checkableButtons.append(self.overlayButton)
        self.checkableQButtonsGroup.addButton(self.overlayButton)

        # fluorescent image color widget
        self.colorButton = pg.ColorButton(self, color=(230,230,230))
        self.colorButton.setFixedHeight(32)
        self.colorButton.setDisabled(True)
        self.colorButton.setToolTip('Fluorescent image color')
        navigateToolBar.addWidget(self.colorButton)

        # Assign mother to bud button
        self.assignMothBudButton = QToolButton(self)
        self.assignMothBudButton.setIcon(QIcon(":assign-motherbud.svg"))
        self.assignMothBudButton.setCheckable(True)
        self.assignMothBudButton.setShortcut('m')
        self.assignMothBudButton.setDisabled(True)
        self.assignMothBudButton.setToolTip(
            'Assign bud to the mother cell.\n'
            'Active only in "Cell cycle analysis" mode (M + right-click)'
        )
        navigateToolBar.addWidget(self.assignMothBudButton)
        self.checkableButtons.append(self.assignMothBudButton)
        self.checkableQButtonsGroup.addButton(self.assignMothBudButton)


        self.navigateToolBar = navigateToolBar



        # Edit toolbar
        editToolBar = QToolBar("Edit", self)
        # editToolBar.setFixedHeight(72)
        editToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(editToolBar)

        self.brushButton = QToolButton(self)
        self.brushButton.setIcon(QIcon(":brush.svg"))
        self.brushButton.setCheckable(True)
        self.brushButton.setShortcut('b')
        self.brushButton.setToolTip('Paint (b)')
        editToolBar.addWidget(self.brushButton)
        self.checkableButtons.append(self.brushButton)

        self.eraserButton = QToolButton(self)
        self.eraserButton.setIcon(QIcon(":eraser.png"))
        self.eraserButton.setCheckable(True)
        self.eraserButton.setShortcut('x')
        self.eraserButton.setToolTip('Erase (x)')
        editToolBar.addWidget(self.eraserButton)
        self.checkableButtons.append(self.eraserButton)

        self.editID_Button = QToolButton(self)
        self.editID_Button.setIcon(QIcon(":edit-id.svg"))
        self.editID_Button.setCheckable(True)
        self.editID_Button.setShortcut('n')
        self.editID_Button.setToolTip('Edit ID (N + right-click)')
        editToolBar.addWidget(self.editID_Button)
        self.checkableButtons.append(self.editID_Button)
        self.checkableQButtonsGroup.addButton(self.editID_Button)

        self.separateBudButton = QToolButton(self)
        self.separateBudButton.setIcon(QIcon(":separate-bud.svg"))
        self.separateBudButton.setCheckable(True)
        self.separateBudButton.setShortcut('s')
        self.separateBudButton.setToolTip(
            'Separate bud (S + right-click)\n'
            'Keep "S" pressed down to enforce manual separation'
        )
        editToolBar.addWidget(self.separateBudButton)
        self.checkableButtons.append(self.separateBudButton)
        self.checkableQButtonsGroup.addButton(self.separateBudButton)

        self.mergeIDsButton = QToolButton(self)
        self.mergeIDsButton.setIcon(QIcon(":merge-IDs.svg"))
        self.mergeIDsButton.setCheckable(True)
        self.mergeIDsButton.setShortcut('m')
        self.mergeIDsButton.setToolTip('Merge IDs (S + right-click)')
        editToolBar.addWidget(self.mergeIDsButton)
        self.checkableButtons.append(self.mergeIDsButton)
        self.checkableQButtonsGroup.addButton(self.mergeIDsButton)

        self.binCellButton = QToolButton(self)
        self.binCellButton.setIcon(QIcon(":bin.svg"))
        self.binCellButton.setCheckable(True)
        self.binCellButton.setToolTip(
           "Annotate cell as 'Removed from analysis' (R + right-click)"
        )
        self.binCellButton.setShortcut("r")
        editToolBar.addWidget(self.binCellButton)
        self.checkableButtons.append(self.binCellButton)
        self.checkableQButtonsGroup.addButton(self.binCellButton)

        self.ripCellButton = QToolButton(self)
        self.ripCellButton.setIcon(QIcon(":rip.svg"))
        self.ripCellButton.setCheckable(True)
        self.ripCellButton.setToolTip(
           "Annotate cell as dead (D + right-click)"
        )
        self.ripCellButton.setShortcut("d")
        editToolBar.addWidget(self.ripCellButton)
        self.checkableButtons.append(self.ripCellButton)
        self.checkableQButtonsGroup.addButton(self.ripCellButton)

        editToolBar.addAction(self.repeatTrackingAction)

        self.disableTrackingCheckBox = QCheckBox("Disable tracking")
        self.disableTrackingCheckBox.setLayoutDirection(Qt.RightToLeft)
        editToolBar.addWidget(self.disableTrackingCheckBox)

        self.brushSizeSpinbox = QSpinBox()
        self.brushSizeSpinbox.setValue(4)
        self.brushSizeLabel = QLabel('   Size: ')
        self.brushSizeLabel.setBuddy(self.brushSizeSpinbox)
        editToolBar.addWidget(self.brushSizeLabel)
        editToolBar.addWidget(self.brushSizeSpinbox)

        # Edit toolbar
        modeToolBar = QToolBar("Mode", self)
        # editToolBar.setFixedHeight(72)
        modeToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(modeToolBar)

        self.modeComboBox = QComboBox()
        self.modeComboBox.addItems(['Segmentation and Tracking',
                                    'Cell cycle analysis',
                                    'Viewer'])
        self.modeComboBoxLabel = QLabel('    Mode: ')
        self.modeComboBoxLabel.setBuddy(self.modeComboBox)
        modeToolBar.addWidget(self.modeComboBoxLabel)
        modeToolBar.addWidget(self.modeComboBox)


        self.modeToolBar = modeToolBar
        self.editToolBar = editToolBar

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
        self.openAction = QAction(QIcon(":folder-open.svg"), "&Open...", self)
        self.saveAction = QAction(QIcon(":file-save.svg"),
                                  "&Save (Ctrl+S)", self)
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
        # Edit actions
        self.repeatSegmActionYeaZ = QAction("YeaZ", self)
        self.repeatSegmActionCellpose = QAction("Cellpose", self)
        self.prevAction = QAction(QIcon(":arrow-left.svg"),
                                        "Previous frame", self)
        self.nextAction = QAction(QIcon(":arrow-right.svg"),
                                  "Next Frame", self)
        self.prevAction.setShortcut("left")
        self.nextAction.setShortcut("right")
        self.repeatTrackingAction = QAction(
            QIcon(":repeat-tracking.svg"), "Repeat tracking", self
        )
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
        self.showInExplorerAction.triggered.connect(self.showInExplorer)
        self.exitAction.triggered.connect(self.close)

        self.undoAction.triggered.connect(self.undo)
        self.redoAction.triggered.connect(self.redo)

        # Connect Help actions
        self.helpContentAction.triggered.connect(self.helpContent)
        self.aboutAction.triggered.connect(self.about)
        # Connect Open Recent to dynamically populate it
        self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)

        self.checkableQButtonsGroup.buttonClicked.connect(self.uncheckQButton)

    def uncheckQButton(self, button):
        # Manual exclusive where we allow to uncheck all buttons
        for b in self.checkableQButtonsGroup.buttons():
            if b != button:
                b.setChecked(False)

    def gui_connectEditActions(self):
        self.setEnabledToolbarButton(enabled=True)
        self.prevAction.triggered.connect(self.prev_cb)
        self.nextAction.triggered.connect(self.next_cb)
        self.overlayButton.toggled.connect(self.overlay_cb)
        self.reloadAction.triggered.connect(self.reload_cb)
        self.slideshowButton.toggled.connect(self.launchSlideshow)
        self.repeatSegmActionYeaZ.triggered.connect(self.repeatSegmYeaZ)
        self.repeatSegmActionCellpose.triggered.connect(self.repeatSegmCellpose)
        self.disableTrackingCheckBox.clicked.connect(self.disableTracking)
        self.repeatTrackingAction.triggered.connect(self.repeatTracking)
        self.brushButton.toggled.connect(self.Brush_cb)
        self.eraserButton.toggled.connect(self.Eraser_cb)
        # Brush/Eraser size action
        self.brushSizeSpinbox.valueChanged.connect(self.brushSize_cb)
        # Mode
        self.modeComboBox.activated[str].connect(self.changeMode)
        self.equalizeHistPushButton.clicked.connect(self.equalizeHist)
        self.colorButton.sigColorChanging.connect(self.updateOverlay)
        self.alphaScrollBar.valueChanged.connect(self.updateOverlay)
        # Drawing mode
        self.drawIDsContComboBox.currentIndexChanged.connect(
                                                self.drawIDsContComboBox_cb)

    def drawIDsContComboBox_cb(self, idx):
        self.updateALLimg()
        how = self.drawIDsContComboBox.currentText()
        onlyIDs = how == 'Draw only IDs'
        nothing = how == 'Draw nothing'
        onlyCont = how == 'Draw only contours'
        only_ccaInfo = how == 'Draw only cell cycle info'
        ccaInfo_and_cont = how == 'Draw cell cycle info and contours'

        t0 = time.time()
        # Clear contours if requested
        if how.find('contours') == -1 or nothing:
            for curveID in self.ax1_ContoursCurves:
                curveID.setData([], [])
            t1 = time.time()

            print(f'Clearing contours = {t1-t0:.3f}')

        t0 = time.time()

        # Clear LabelItems IDs if requested (draw nothing or only contours)
        if onlyCont or nothing:
            for _IDlabel1 in self.ax1_LabelItemsIDs:
                _IDlabel1.setText('')
            t1 = time.time()

            print(f'Clearing labels = {t1-t0:.3f}')

    def setEnabledToolbarButton(self, enabled=False):
        self.showInExplorerAction.setEnabled(enabled)
        self.reloadAction.setEnabled(enabled)
        self.saveAction.setEnabled(enabled)
        self.editToolBar.setEnabled(enabled)
        self.navigateToolBar.setEnabled(enabled)
        self.modeToolBar.setEnabled(enabled)
        self.enableSizeSpinbox(False)
        if not enabled:
            self.setUncheckedAllButtons()

    def enableSizeSpinbox(self, enabled):
        self.brushSizeLabel.setEnabled(enabled)
        self.brushSizeSpinbox.setEnabled(enabled)

    def reload_cb(self):
        self.app.setOverrideCursor(Qt.WaitCursor)
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)
        self.lab = np.load(self.data.segm_npy_path)[self.frame_i]
        self.update_rp()
        self.updateALLimg()
        self.app.restoreOverrideCursor()

    def changeMode(self, mode):
        if mode == 'Segmentation and Tracking':
            self.setEnabledToolbarButton(enabled=True)
            self.disableTrackingCheckBox.setChecked(False)
            self.assignMothBudButton.setDisabled(True)
            self.drawIDsContComboBox.clear()
            self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxSegmItems)
            for BudMothLine in self.ax1_BudMothLines:
                BudMothLine.setData([], [])
        elif mode == 'Cell cycle analysis':
            proceed = self.init_cca()
            if proceed:
                self.setEnabledToolbarButton(enabled=False)
                self.showInExplorerAction.setEnabled(True)
                self.saveAction.setEnabled(True)
                self.navigateToolBar.setEnabled(True)
                self.modeToolBar.setEnabled(True)
                self.disableTrackingCheckBox.setChecked(True)
                self.assignMothBudButton.setDisabled(False)
                self.drawIDsContComboBox.clear()
                self.drawIDsContComboBox.addItems(
                                        self.drawIDsContComboBoxCcaItems)

        elif mode == 'Viewer':
            self.setEnabledToolbarButton(enabled=False)
            self.showInExplorerAction.setEnabled(True)
            self.navigateToolBar.setEnabled(True)
            self.modeToolBar.setEnabled(True)
            self.disableTrackingCheckBox.setChecked(True)
            self.undoAction.setDisabled(True)
            self.redoAction.setDisabled(True)
            currentMode = self.drawIDsContComboBox.currentText()
            self.drawIDsContComboBox.clear()
            self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxCcaItems)
            self.drawIDsContComboBox.setCurrentText(currentMode)


    def launchSlideshow(self):
        if self.slideshowButton.isChecked():
            self.slideshowWin = apps.CellsSlideshow_GUI(
                                   button_toUncheck=self.slideshowButton,
                                   Left=self.slideshowWinLeft,
                                   Top=self.slideshowWinTop)
            self.slideshowWin.loadData(self.data.img_data, frame_i=self.frame_i)
            self.slideshowWin.show()
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

    def brushSize_cb(self):
        self.EraserCircle.setSize(self.brushSizeSpinbox.value()*2)
        self.ax1_BrushCircle.setSize(self.brushSizeSpinbox.value()*2)
        self.ax2_BrushCircle.setSize(self.brushSizeSpinbox.value()*2)


    def Brush_cb(self, event):
        # Toggle eraser Button OFF
        if self.eraserButton.isChecked():
            self.eraserButton.toggled.disconnect()
            self.eraserButton.setChecked(False)
            self.eraserButton.toggled.connect(self.Eraser_cb)

        if not self.brushButton.isChecked():
            self.ax1_BrushCircle.setData([], [])
            self.ax2_BrushCircle.setData([], [])
            self.enableSizeSpinbox(False)
        else:
            if self.img2.image.max()+1 > self.brushID:
                self.brushID = self.img2.image.max()+1
            else:
                self.brushID += 1
            self.enableSizeSpinbox(True)
            try:
                self.gui_hoverEventImg1(self.hoverEventImg1)
            except:
                pass

    def equalizeHist(self):
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)
        img = skimage.exposure.equalize_adapthist(self.img1.image)
        self.img1.setImage(img)


    def Eraser_cb(self, event):
        if self.brushButton.isChecked():
            self.brushButton.toggled.disconnect()
            self.brushButton.setChecked(False)
            self.brushButton.toggled.connect(self.Brush_cb)
        if not self.eraserButton.isChecked():
            self.ax1_BrushCircle.setData([], [])
            self.ax2_BrushCircle.setData([], [])
            self.brushButton.setChecked(False)
            self.enableSizeSpinbox(False)
        else:
            self.enableSizeSpinbox(True)

    def keyPressEvent(self, ev):
        isBrushActive = (self.brushButton.isChecked()
                      or self.eraserButton.isChecked())
        if ev.key() == Qt.Key_Up and isBrushActive:
            self.brushSizeSpinbox.setValue(self.brushSizeSpinbox.value()+1)
        elif ev.key() == Qt.Key_Down and isBrushActive:
            self.brushSizeSpinbox.setValue(self.brushSizeSpinbox.value()-1)
        elif ev.key() == Qt.Key_Escape:
            self.setUncheckedAllButtons()
        elif ev.key() == Qt.Key_Alt:
            self.app.setOverrideCursor(Qt.SizeAllCursor)
            self.isAltDown = True
        elif ev.modifiers() == Qt.ControlModifier:
            if ev.key() == Qt.Key_P:
                print('------------------------')
                print('Cell cycle analysis table:')
                print(self.cca_df)
                print('------------------------')
        # elif ev.key() == Qt.Key_Plus:
        #     print('Programmatically disabled')
        #     self.disableTrackingCheckBox.setChecked(True)
        elif ev.key() == Qt.Key_H:
            lab_mask = (self.lab>0).astype(np.uint8)
            rp = skimage.measure.regionprops(lab_mask)
            obj = rp[0]
            min_row, min_col, max_row, max_col = obj.bbox
            xRange = min_col-10, max_col+10
            yRange = max_row+10, min_row-10
            self.ax1.setRange(xRange=xRange, yRange=yRange)

        # elif ev.text() == 'b':
        #     self.BrushEraser_cb(ev)

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key_Alt:
            self.isAltDown = False
            self.app.restoreOverrideCursor()

    def setUncheckedAllButtons(self):
        self.draw_MothBudTempLine = False
        try:
            self.BudMothTempLine.setData([], [])
        except:
            pass
        for button in self.checkableButtons:
            button.setChecked(False)

    def propagateChange(self, modID, modTxt, doNotShow, UndoFutFrames,
                        applyFutFrames, applyTrackingB=False):
        """
        This function determines whether there are already visited future frames
        that contains "modID". If so, it triggers a pop-up asking the user
        what to do (propagate change to future frames o not)
        """

        areFutureIDs_affected = []
        # Get number of future frames already visited and checked if future
        # frames has an ID affected by the change
        for i in range(self.frame_i+1, self.num_segm_frames):
            if self.allData_li[i]['labels'] is None:
                break
            else:
                futureIDs = np.unique(self.allData_li[i]['labels'])
                if modID in futureIDs:
                    areFutureIDs_affected.append(True)

        if i == self.frame_i+1:
            # No future frames to propagate the change to
            return False, False, None, doNotShow

        if not areFutureIDs_affected:
            # There are future frames but they are not affected by the change
            return UndoFutFrames, applyFutFrames, None, doNotShow

        # Ask what to do unless the user has previously checked doNotShowAgain
        if doNotShow:
            endFrame_i = i
        else:
            ffa = apps.FutureFramesAction_QDialog(
                    self.frame_i+1, i, modTxt, applyTrackingB=applyTrackingB)
            ffa.exec_()
            decision = ffa.decision
            endFrame_i = ffa.endFrame_i
            doNotShowAgain = ffa.doNotShowCheckbox.isChecked()

            if decision is None:
                return None, None, None, doNotShow

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



    def addCurrentState(self):
        self.UndoRedoStates[self.frame_i].insert(
                           0, {'image': self.img1.image.copy(),
                               'labels': self.lab.copy(),
                               'editID_info': self.editID_info.copy(),
                               'binnedIDs':self.binnedIDs.copy(),
                               'ripIDs':self.ripIDs.copy()}
        )

    def getCurrentState(self):
        i = self.frame_i
        c = self.UndoCount
        self.cells_img = self.UndoRedoStates[i][c]['image'].copy()
        self.lab = self.UndoRedoStates[i][c]['labels'].copy()
        self.editID_info = self.UndoRedoStates[i][c]['editID_info'].copy()
        self.binnedIDs = self.UndoRedoStates[i][c]['binnedIDs'].copy()
        self.ripIDs = self.UndoRedoStates[i][c]['ripIDs'].copy()

    def storeUndoRedoStates(self, UndoFutFrames):

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
        if len(self.UndoRedoStates[self.frame_i]) > 5:
            self.UndoRedoStates[self.frame_i].pop(-1)

    def undo(self):
        if self.UndoCount == 0:
            # Store current state to enable redoing it
            self.addCurrentState()

        # Get previously stored state
        if self.UndoCount < len(self.UndoRedoStates[self.frame_i])-1:
            self.UndoCount += 1
            # Since we have undone then it is possible to redo
            self.redoAction.setEnabled(True)

            # Restore state
            self.getCurrentState()
            self.update_rp()
            self.checkIDs_LostNew()
            self.updateALLimg(image=self.cells_img)

        if not self.UndoCount < len(self.UndoRedoStates[self.frame_i])-1:
            # We have undone all available states
            self.undoAction.setEnabled(False)



    def redo(self):
        # Get previously stored state
        if self.UndoCount > 0:
            self.UndoCount -= 1
            # Since we have redone then it is possible to undo
            self.undoAction.setEnabled(True)

            # Restore state
            self.getCurrentState()
            self.update_rp()
            self.checkIDs_LostNew()
            self.updateALLimg(image=self.cells_img)

        if not self.UndoCount > 0:
            # We have redone all available states
            self.redoAction.setEnabled(False)

    def disableTracking(self, isChecked):
        # Event called ONLY if the user click on Disable tracking
        # NOT called if setChecked is called. This allows to keep track
        # of the user choice. This way user con enforce tracking
        # NOTE: I know two booleans doing the same thing is overkill
        # but the code is more readable when we actually need them
        if self.disableTrackingCheckBox.isChecked():
            self.UserEnforced_DisabledTracking = True
            self.UserEnforced_Tracking = False
        else:
            warn_txt = (

            'You requested to explicitly ENABLE tracking. This will '
            'overwrite the default behaviour of not tracking already '
            'visited/checked frames.\n '
            'On all future frames that you will visit tracking '
            'will be automatically performed unless you explicitly '
            'disable tracking by clicking "Disable tracking" again.\n\n'
            'If you need to repeat tracking ONLY on the current frame you '
            'can use the "Repeat tracking" button on the toolbar instead.\n\n'
            'Are you sure you want to proceed with ENABLING tracking from now on?'

            )
            msg = QtGui.QMessageBox()
            enforce_Tracking = msg.warning(
               self, 'Disable tracking?', warn_txt, msg.Yes | msg.No
            )
            if enforce_Tracking == msg.No:
                self.disableTrackingCheckBox.setChecked(True)
            else:
                self.repeatTracking()
                self.UserEnforced_DisabledTracking = False
                self.UserEnforced_Tracking = True

    def repeatTracking(self):
        self.tracking(enforce=True, DoManualEdit=False)
        if self.editID_info:
            editIDinfo = [
                f'Replace ID {self.lab[y,x]} with {newID}'
                for y, x, newID in self.editID_info
            ]
            msg = QtGui.QMessageBox()
            msg.setIcon(msg.Information)
            msg.setText("You requested to repeat tracking but there are "
                        "the following manually editied IDs:\n\n"
                        f"{editIDinfo}\n\n"
                        "Do you want to keep these edits or ignore them?")
            keepManualEditButton = QPushButton('Keep manually edited IDs')
            msg.addButton(keepManualEditButton, msg.YesRole)
            msg.addButton(QPushButton('Ignore'), msg.NoRole)
            msg.exec_()
            if msg.clickedButton() == keepManualEditButton:
                allIDs = [obj.label for obj in self.rp]
                self.ManuallyEditTracking(self.lab, allIDs)
                self.update_rp()
                self.checkIDs_LostNew()
            else:
                self.editID_info = []

        self.updateALLimg()

    def repeatSegmYeaZ(self):
        t0 = time.time()
        self.which_model = 'YeaZ'
        if self.is_first_call_YeaZ:
            print('Importing YeaZ model...')
            from YeaZ.unet import neural_network as nn
            from YeaZ.unet import segment
            self.nn = nn
            self.segment = segment
            self.path_weights = nn.determine_path_weights()
            download_model('YeaZ')

        img = skimage.exposure.equalize_adapthist(self.img1.image)
        pred = self.nn.prediction(img, is_pc=True,
                                  path_weights=self.path_weights)
        thresh = self.nn.threshold(pred)
        lab = self.segment.segment(thresh, pred, min_distance=5).astype(int)
        self.is_first_call_YeaZ = False
        self.lab = lab
        self.update_rp()
        self.tracking()
        self.updateALLimg()
        t1 = time.time()
        print('-----------------')
        print(f'Segmentation done in {t1-t0:.3f} s')
        print('=================')

    def repeatSegmCellpose(self):
        t0 = time.time()
        self.which_model = 'Cellpose'
        if self.is_first_call_cellpose:
            print('Initializing cellpose models...')
            from acdc_cellpose import models
            download_model('cellpose')
            device, gpu = models.assign_device(True, False)
            self.cp_model = models.Cellpose(gpu=gpu, device=device,
                                            model_type='cyto', torch=True)

        img = skimage.exposure.equalize_adapthist(self.img1.image)
        apps.imshow_tk(img)
        lab, flows, _, _ = self.cp_model.eval(img, channels=[0,0],
                                                   diameter=60,
                                                   invert=False,
                                                   net_avg=True,
                                                   augment=False,
                                                   resample=False,
                                                   do_3D=False,
                                                   progress=None)
        self.is_first_call_cellpose = False
        self.lab = lab
        self.update_rp()
        self.tracking()
        self.updateALLimg()
        t1 = time.time()
        print('-----------------')
        print(f'Segmentation done in {t1-t0:.3f} s')
        print('=================')

    def next_cb(self):
        mode = str(self.modeComboBox.currentText())
        if self.frame_i < self.num_segm_frames-1:
            if 'lost' in self.titleLabel.text:
                msg = QtGui.QMessageBox()
                warn_msg = (
                    'Current frame (compared to previous frame) '
                    'has lost the following cells:\n\n'
                    f'{self.lost_IDs}\n\n'
                    'Are you sure you want to continue?'
                )
                proceed_with_lost = msg.warning(
                   self, 'Lost cells!', warn_msg, msg.Yes | msg.No
                )
                if proceed_with_lost == msg.No:
                    return

            if self.frame_i <= 0 and mode == 'Cell cycle analysis':
                IDs = [obj.label for obj in self.rp]
                init_cca_df_frame0 = apps.cca_df_frame0(IDs, self.cca_df)
                if init_cca_df_frame0.cancel:
                    return
                self.cca_df = init_cca_df_frame0.df
                self.store_cca_df()

            # Store data for current frame
            self.store_data(debug=False)
            # Go to next frame
            self.frame_i += 1
            proceed_cca, never_visited = self.get_data()
            if not proceed_cca:
                self.frame_i -= 1
                self.get_data()
                self.update_rp_metadata(draw=False)
                return
            self.tracking(storeUndo=True)
            notEnoughG1Cells, proceed = self.attempt_auto_cca()
            if notEnoughG1Cells or not proceed:
                self.frame_i -= 1
                self.get_data()
                self.update_rp_metadata(draw=False)
                return
            self.updateALLimg(never_visited=never_visited)
        else:
            msg = 'You reached the last segmented frame!'
            print(msg)
            self.titleLabel.setText(msg, color='w')
        if self.slideshowWin is not None:
            self.slideshowWin.frame_i = self.frame_i
            self.slideshowWin.update_img()

    def prev_cb(self):
        if self.frame_i > 0:
            self.store_data()
            self.frame_i -= 1
            _, never_visited = self.get_data()
            self.tracking()
            self.updateALLimg(never_visited=never_visited)
        else:
            msg = 'You reached the first frame!'
            print(msg)
            self.titleLabel.setText(msg, color='w')
        if self.slideshowWin is not None:
            self.slideshowWin.frame_i = self.frame_i
            self.slideshowWin.update_img()

    def init_frames_data(self, frames_path, user_ch_name):
        data = load.load_frames_data(frames_path, user_ch_name)
        # Allow single 2D/3D image
        if data.SizeT < 2:
            data.img_data = np.array([data.img_data])
            data.segm_data = np.array([data.segm_data])
        img_shape = data.img_data.shape
        self.num_frames = len(data.img_data)
        self.num_segm_frames = len(data.segm_data)
        SizeT = data.SizeT
        SizeZ = data.SizeZ
        print(f'Data shape = {img_shape}')
        print(f'Number of frames = {SizeT}')
        print(f'Number of z-slices per frame = {SizeZ}')

        self.data = data

        self.gui_createGraphicsItems()

        self.init_attr(max_ID=data.segm_data.max())

    def clear_prevItems(self):
        # Clear data from those items that have data and are not present in
        # current rp
        IDs = [obj.label for obj in self.rp]
        allItems = zip(self.ax1_ContoursCurves,
                       self.ax1_LabelItemsIDs,
                       self.ax2_LabelItemsIDs,
                       self.ax1_BudMothLines)
        for idx, items_ID in enumerate(allItems):
            ContCurve, _IDlabel1, _IDlabel2, BudMothLine = items_ID
            ID = idx+1

            if ContCurve.getData()[0] is not None and ID not in IDs:
                # Contour is present but ID is not --> clear
                ContCurve.setData([], [])

            if _IDlabel1.text != '' and ID not in IDs:
                # Text is present but ID is not --> clear
                _IDlabel1.setText('')

            if _IDlabel2.text != '' and ID not in IDs:
                # Text is present but ID is not --> clear
                _IDlabel2.setText('')

            if BudMothLine.getData()[0] is not None and ID not in IDs:
                # Contour is present but ID is not --> clear
                BudMothLine.setData([], [])

    def init_attr(self, max_ID=10):
        # Decision on what to do with changes to future frames attr
        self.doNotShowAgain_EditID = False
        self.UndoFutFrames_EditID = False
        self.applyFutFrames_EditID = False

        self.doNotShowAgain_RipID = False
        self.UndoFutFrames_RipID = False
        self.applyFutFrames_RipID = False

        self.doNotShowAgain_DelID = False
        self.UndoFutFrames_DelID = False
        self.applyFutFrames_DelID = False

        self.doNotShowAgain_BinID = False
        self.UndoFutFrames_BinID = False
        self.applyFutFrames_BinID = False


        self.draw_MothBudTempLine = False
        self.disableAutoActivateViewerWindow = False
        self.enforceSeparation = False
        self.isAltDown = False
        self.UserEnforced_DisabledTracking = False
        self.UserEnforced_Tracking = False
        self.new_IDs = []
        self.UndoRedoStates = [[] for _ in range(self.num_frames)]

        # Colormap
        self.cmap = pg.colormap.get('viridis', source='matplotlib')
        self.lut = self.cmap.getLookupTable(0,1, max_ID+10)
        np.random.shuffle(self.lut)
        # Insert background color
        self.lut = np.insert(self.lut, 0, [25, 25, 25], axis=0)

        # Plots items
        self.is_first_call_YeaZ = True
        self.is_first_call_cellpose = True
        self.data_loaded = True
        self.isMouseDragImg2 = False

        self.allData_li = [
                {
                 'regionprops': [],
                 'labels': None,
                 'segm_metadata_df': None
                 }
                for i in range(self.num_frames)
        ]
        self.frame_i = 0
        self.brushID = 0
        self.binnedIDs = set()
        self.ripIDs = set()
        self.cca_df = None
        self.cca_df_colnames = ['cell_cycle_stage',
                                'generation_num',
                                'relative_ID',
                                'relationship',
                                'emerg_frame_i',
                                'division_frame_i']
        self.cca_df_int_cols = ['generation_num',
                                'relative_ID',
                                'emerg_frame_i',
                                'division_frame_i']
        if self.data.last_tracked_i is not None:
            last_tracked_num = self.data.last_tracked_i+1
            # Load previous session data
            for i in range(self.data.last_tracked_i):
                self.frame_i = i
                self.get_data()
                self.update_rp_metadata(draw=False)
                self.store_data()
                self.binnedIDs = set()
                self.ripIDs = set()

            # Ask whether to resume from last frame
            msg = QtGui.QMessageBox()
            start_from_last_tracked_i = msg.question(
                self, 'Start from last session?',
                'The system detected a previous session ended '
                f'at frame {last_tracked_num}.\n\n'
                f'Do you want to resume from frame {last_tracked_num}?',
                msg.Yes | msg.No
            )

            if start_from_last_tracked_i == msg.Yes:
                self.frame_i = self.data.last_tracked_i-1
                self.next_cb()
            else:
                self.frame_i = 0
                self.get_data()
                self.IDs = [obj.label for obj in self.rp]
                self.updateALLimg()
        else:
            self.frame_i = 0
            self.get_data()
            self.IDs = [obj.label for obj in self.rp]
            self.updateALLimg()

        # Link Y and X axis of both plots to scroll zoom and pan together
        self.ax2.vb.setYLink(self.ax1.vb)
        self.ax2.vb.setXLink(self.ax1.vb)

        self.ax2.vb.autoRange()
        self.ax1.vb.autoRange()

    def store_data(self, debug=False):
        if self.frame_i < 0:
            # In some cases we set frame_i = -1 and then call next_cb
            # to visualize frame 0. In that case we don't store data
            # for frame_i = -1
            return

        mode = str(self.modeComboBox.currentText())

        self.allData_li[self.frame_i]['regionprops'] = self.rp.copy()
        self.allData_li[self.frame_i]['labels'] = self.lab.copy()

        if debug:
            pass
            # apps.imshow_tk(self.lab)

        # Store dynamic metadata
        is_cell_dead_li = [False]*len(self.rp)
        is_cell_excluded_li = [False]*len(self.rp)
        IDs = [0]*len(self.rp)
        xx_centroid = [0]*len(self.rp)
        yy_centroid = [0]*len(self.rp)
        editIDclicked_x = [np.nan]*len(self.rp)
        editIDclicked_y = [np.nan]*len(self.rp)
        editIDnewID = [-1]*len(self.rp)
        editedIDs = [newID for _, _, newID in self.editID_info]
        for i, obj in enumerate(self.rp):
            is_cell_dead_li[i] = obj.dead
            is_cell_excluded_li[i] = obj.excluded
            IDs[i] = obj.label
            xx_centroid[i] = int(round(obj.centroid[1]))
            yy_centroid[i] = int(round(obj.centroid[0]))
            if obj.label in editedIDs:
                y, x, new_ID = self.editID_info[editedIDs.index(obj.label)]
                editIDclicked_x[i] = int(round(x))
                editIDclicked_y[i] = int(round(y))
                editIDnewID[i] = new_ID

        self.allData_li[self.frame_i]['segm_metadata_df'] = pd.DataFrame(
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

        self.store_cca_df()

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
        min_dist = dist.min()
        return min_dist, nearest_point

    def attempt_auto_cca(self):
        doNotProceed = False
        try:
            notEnoughG1Cells, proceed = self.autoCca_df()
        except:
            traceback.print_exc()
            msg = QtGui.QMessageBox()
            warn_cca = msg.critical(
                self, 'Failed cell cycle analysis',
                f'Cell cycle analysis for frame {self.frame_i+1} failed!\n\n'
                'This is most likely because the next frame has '
                'segmentation or tracking errors.\n\n'
                'Switch to "Segmentation and Tracking" mode and '
                'check/correct next frame,\n'
                'before attempting cell cycle analysis again.\n\n'
                'NOTE: See console for details on the error occured.',
                msg.Ok
            )
        return notEnoughG1Cells, proceed

    def autoCca_df(self):
        """
        Assign each bud to a mother with scipy linear sum assignment
        (Hungarian or Munkres algorithm). First we build a cost matrix where
        each (i, j) element is the minimum distance between bud i and mother j.
        Then we minimize the cost of assigning each bud to a mother, and finally
        we write the assignment info into cca_df
        """
        proceed = True
        notEnoughG1Cells = False

        # Skip cca if not the right mode
        mode = str(self.modeComboBox.currentText())
        if mode.find('Cell cycle') == -1:
            return notEnoughG1Cells, proceed

        # Use stored cca_df and do not modify it with automatic stuff
        if self.cca_df is not None:
            return notEnoughG1Cells, proceed

        # Make sure that this is a visited frame
        df = self.allData_li[self.frame_i-1]['segm_metadata_df']
        if df is None or 'cell_cycle_stage' not in df.columns:
            msg = QtGui.QMessageBox()
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
            proceed = False
            return notEnoughG1Cells, proceed

        self.cca_df = df[self.cca_df_colnames].copy()

        # If there are no new IDs we are done
        if not self.new_IDs:
            proceed = True
            return notEnoughG1Cells, proceed

        # Check if there are enough cells in G1
        df_G1 = self.cca_df[self.cca_df['cell_cycle_stage']=='G1']
        IDsCellsG1 = list(df_G1.index)
        numCellsG1 = len(df_G1)
        numNewCells = len(self.new_IDs)
        if numCellsG1 < numNewCells:
            msg = QtGui.QMessageBox()
            warn_cca = msg.critical(
                self, 'No cells in G1!',
                f'In the next frame {numNewCells} new buds will '
                'emerge.\n'
                f'However there are only {numCellsG1} cells '
                'in G1 available.\n\n'
                'Switch to "Segmentation and Tracking" mode '
                'and check/correct next frame,\n'
                'before attempting cell cycle analysis again',
                msg.Ok
            )
            notEnoughG1Cells = True
            proceed = False
            return notEnoughG1Cells, proceed

        # Compute new IDs contours
        newIDs_contours = []
        for obj in self.rp:
            ID = obj.label
            if ID in self.new_IDs:
                cont = self.get_objContours(obj)
                newIDs_contours.append(cont)

        # Compute cost matrix
        cost = np.zeros((numCellsG1, numNewCells))
        for obj in self.rp:
            ID = obj.label
            if ID in IDsCellsG1:
                cont = self.get_objContours(obj)
                i = IDsCellsG1.index(ID)
                for j, newID_cont in enumerate(newIDs_contours):
                    min_dist, nearest_xy = self.nearest_point_2Dyx(
                                                         cont, newID_cont)
                    cost[i, j] = min_dist

        # Run hungarian (munkres) assignment algorithm
        row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost)

        # Assign buds to mothers
        for i, j in zip(row_idx, col_idx):
            mothID = IDsCellsG1[i]
            budID = self.new_IDs[j]
            self.cca_df.at[mothID, 'relative_ID'] = ID
            self.cca_df.at[mothID, 'cell_cycle_stage'] = 'S'

            self.cca_df.loc[budID] = {
                'cell_cycle_stage': 'S',
                'generation_num': 0,
                'relative_ID': mothID,
                'relationship': 'bud',
                'emerg_frame_i': self.frame_i,
                'division_frame_i': -1
            }

        self.store_cca_df()
        proceed = True
        return notEnoughG1Cells, proceed


    def get_objContours(self, obj):
        contours, _ = cv2.findContours(
                               obj.image.astype(np.uint8),
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE
        )
        min_y, min_x, _, _ = obj.bbox
        cont = np.squeeze(contours[0], axis=1)
        cont = np.vstack((cont, cont[0]))
        cont += [min_x, min_y]
        return cont

    def get_data(self):
        proceed_cca = True
        if self.frame_i > 2:
            # Remove undo states from 2 frames back to avoid memory issues
            self.UndoRedoStates[self.frame_i-2] = []
            # Check if current frame contains undo states (not empty list)
            if self.UndoRedoStates[self.frame_i]:
                self.undoAction.setDisabled(False)
            else:
                self.undoAction.setDisabled(True)
        self.UndoCount = 0
        # If stored labels is None then it is the first time we visit this frame
        if self.allData_li[self.frame_i]['labels'] is None:
            self.editID_info = []
            never_visited = True
            if str(self.modeComboBox.currentText()) == 'Cell cycle analysis':
                # Warn that we are visiting a frame that was never segm-checked
                # on cell cycle analysis mode
                msg = QtGui.QMessageBox()
                warn_cca = msg.critical(
                    self, 'Never checked segmentation on requested frame',
                    'Segmentation and Tracking was never checked from '
                    f'frame {self.frame_i+1} onward.\n To ensure correct cell '
                    'cell cycle analysis you have to first visit frames '
                    f'{self.frame_i+1}-end with "Segmentation and Tracking" mode.',
                    msg.Ok
                )
                proceed_cca = False
                return proceed_cca, never_visited
            # Requested frame was never visited before. Load from HDD
            self.lab = self.data.segm_data[self.frame_i].copy()
            self.rp = skimage.measure.regionprops(self.lab)
            if self.data.segm_metadata_df is not None:
                frames = self.data.segm_metadata_df.index.get_level_values(0)
                if self.frame_i in frames:
                    # Since there was already segmentation metadata from
                    # previous closed session add it to current metadata
                    df = self.data.segm_metadata_df.loc[self.frame_i].copy()
                    binnedIDs_df = df[df['is_cell_excluded']]
                    binnedIDs = set(binnedIDs_df.index).union(self.binnedIDs)
                    self.binnedIDs = binnedIDs
                    ripIDs_df = df[df['is_cell_dead']]
                    ripIDs = set(ripIDs_df.index).union(self.ripIDs)
                    self.ripIDs = ripIDs
                    # Load cca df into current metadata
                    if 'cell_cycle_stage' in df.columns:
                        if any(df['cell_cycle_stage'].isna()):
                            df = df.drop(labels=self.cca_df_colnames, axis=1)
                        else:
                            # Convert to ints since there were NaN
                            cols = self.cca_df_int_cols
                            df[cols] = df[cols].astype(int)
                    i = self.frame_i
                    self.allData_li[i]['segm_metadata_df'] = df.copy()

            self.get_cca_df()
        else:
            # Requested frame was already visited. Load from RAM.
            never_visited = False
            self.lab = self.allData_li[self.frame_i]['labels'].copy()
            self.rp = skimage.measure.regionprops(self.lab)
            df = self.allData_li[self.frame_i]['segm_metadata_df']
            binnedIDs_df = df[df['is_cell_excluded']]
            self.binnedIDs = set(binnedIDs_df.index)
            ripIDs_df = df[df['is_cell_dead']]
            self.ripIDs = set(ripIDs_df.index)
            editIDclicked_x = df['editIDclicked_x'].to_list()
            editIDclicked_y = df['editIDclicked_y'].to_list()
            editIDnewID = df['editIDnewID'].to_list()
            _zip = zip(editIDclicked_y, editIDclicked_x, editIDnewID)
            self.editID_info = [
                (int(y),int(x),newID) for y,x,newID in _zip if newID!=-1]
            self.get_cca_df()

        return proceed_cca, never_visited

    def init_cca(self):
        if self.data.last_tracked_i is None:
            txt = (
                'On this dataset you never checked that the segmentation '
                'and tracking are correct.\n'
                'You first have to check (and eventually correct) some frames'
                'in "Segmentation and Tracking" mode before proceeding '
                'with cell cycle analysis.')
            msg = QtGui.QMessageBox()
            msg.critical(
                self, 'Tracking check not performed', txt, msg.Ok
            )
            return

        proceed = True
        last_cca_frame_i = 0
        # Determine last annotated frame index
        for i, dict_frame_i in enumerate(self.allData_li):
            df = dict_frame_i['segm_metadata_df']
            if df is not None:
                if 'cell_cycle_stage' not in df.columns:
                    last_cca_frame_i = i-1
                    break

        if last_cca_frame_i < 0:
            last_cca_frame_i = 0

        if self.frame_i > last_cca_frame_i:
            # Prompt user to go to last annotated frame
            msg = QtGui.QMessageBox()
            goTo_last_annotated_frame_i = msg.warning(
                self, 'Go to last annotated frame?',
                f'The last annotated frame is frame {last_cca_frame_i+1}.\n'
                'The cell cycle analysis will restart from that frame.\n'
                'Do you want to proceed?',
                msg.Yes | msg.Cancel
            )
            if goTo_last_annotated_frame_i == msg.Yes:
                self.last_cca_frame_i = last_cca_frame_i
                self.frame_i = last_cca_frame_i
                self.get_data()
                self.updateALLimg()
            else:
                print('Cell cycle analysis aborted.')
                proceed = False
                return
        elif self.frame_i < last_cca_frame_i:
            # Prompt user to go to last annotated frame
            msg = QtGui.QMessageBox()
            goTo_last_annotated_frame_i = msg.warning(
                self, 'Go to last annotated frame?',
                f'The last annotated frame is frame {last_cca_frame_i+1}.\n'
                'Do you want to restart cell cycle analysis from frame '
                f'{last_cca_frame_i+1}?',
                msg.Yes | msg.No | msg.Cancel
            )
            if goTo_last_annotated_frame_i == msg.Yes:
                self.last_cca_frame_i = last_cca_frame_i
                self.frame_i = last_cca_frame_i
                self.get_data()
                self.updateALLimg()
            elif goTo_last_annotated_frame_i == msg.Cancel:
                print('Cell cycle analysis aborted.')
                proceed = False
                return
        self.last_cca_frame_i = last_cca_frame_i

        if self.cca_df is None:
            IDs = [obj.label for obj in self.rp]
            cc_stage = ['G1' for ID in IDs]
            num_cycles = [-1]*len(IDs)
            relationship = ['mother' for ID in IDs]
            related_to = [-1]*len(IDs)
            self.cca_df = pd.DataFrame({
                               'cell_cycle_stage': cc_stage,
                               'generation_num': num_cycles,
                               'relative_ID': IDs,
                               'relationship': relationship,
                               'emerg_frame_i': num_cycles,
                               'division_frame_i': num_cycles},
                                index=IDs)
            self.cca_df.index.name = 'Cell_ID'
        else:
            self.get_cca_df()
        return proceed

    def del_future_cca_df(self, from_frame_i):
        for i in range(from_frame_i, self.num_segm_frames):
            df = self.allData_li[i]['segm_metadata_df']
            if df is None:
                # No more saved info to delete
                return

            if 'cell_cycle_stage' not in df.columns:
                # No cell cycle info present
                continue

            df.drop(self.cca_df_colnames, axis=1, inplace=True)





    def get_cca_df(self, frame_i=None, return_df=False):
        # cca_df is None unless the metadata contains cell cycle annotations
        # NOTE: cell cycle annotations are either from the current session
        # or loaded from HDD in "init_attr" with a .question to the user
        cca_df = None
        i = self.frame_i if frame_i is None else frame_i
        df = self.allData_li[i]['segm_metadata_df']
        if df is not None:
            if 'cell_cycle_stage' in df.columns:
                cca_df = df[self.cca_df_colnames].copy()
        if return_df:
            return cca_df
        else:
            self.cca_df = cca_df

    def store_cca_df(self, frame_i=None, cca_df=None):
        i = self.frame_i if frame_i is None else frame_i

        if cca_df is None:
            cca_df = self.cca_df

        if cca_df is not None:
            segm_df = self.allData_li[i]['segm_metadata_df']
            if 'cell_cycle_stage' in segm_df.columns:
                # Cell cycle info already present --> overwrite with new
                df = segm_df
                df[self.cca_df_colnames] = cca_df
            else:
                df = segm_df.join(cca_df, how='outer')
            self.allData_li[i]['segm_metadata_df'] = df.copy()
            # print(self.allData_li[self.frame_i]['segm_metadata_df'])


    def ax1_setTextID(self, obj, how):
        # Draw ID label on ax1 image depending on how
        LabelItemID = self.ax1_LabelItemsIDs[obj.label-1]
        ID = obj.label
        df = self.cca_df
        if df is None or how.find('cell cycle') == -1:
            txt = f'{ID}'
            if ID in self.new_IDs:
                color = 'r'
                bold = True
            else:
                color = 'w'
                bold = False
        else:
            df_ID = df.loc[ID]
            ccs = df_ID['cell_cycle_stage']
            relationship = df_ID['relationship']
            generation_num = df_ID['generation_num']
            generation_num = 'ND' if generation_num==-1 else generation_num
            emerg_frame_i = df_ID['emerg_frame_i']
            is_bud = relationship == 'bud'
            is_moth = relationship == 'mother'
            emerged_now = emerg_frame_i == self.frame_i
            txt = f'{ccs}-{generation_num}'
            if ccs == 'G1':
                c = 0.6
                alpha = 0.7
                color = (255*c, 255*c, 255*c, 255*alpha)
                bold = False
            elif ccs == 'S' and is_moth and not emerged_now:
                color = 0.8
                bold = False
            elif ccs == 'S' and is_bud and not emerged_now:
                color = 'r'
                bold = False
            elif ccs == 'S' and is_bud and emerged_now:
                color = 'r'
                bold = True

        LabelItemID.setText(txt, color=color, bold=bold, size='10pt')

        # Center LabelItem at centroid
        y, x = obj.centroid
        w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
        LabelItemID.setPos(x-w/2, y-h/2)

    def ax2_setTextID(self, obj, how):
        # Draw ID label on ax1 image depending on how
        LabelItemID = self.ax2_LabelItemsIDs[obj.label-1]
        ID = obj.label
        df = self.cca_df
        txt = f'{ID}'
        if ID in self.new_IDs:
            color = 'r'
            bold = True
        else:
            color = (230, 0, 0, 255*0.5)
            bold = False

        LabelItemID.setText(txt, color=color, bold=bold, size='10pt')

        # Center LabelItem at centroid
        y, x = obj.centroid
        w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
        LabelItemID.setPos(x-w/2, y-h/2)


    def drawID_and_Contour(self, obj, drawContours=True):
        how = self.drawIDsContComboBox.currentText()
        IDs_and_cont = how == 'Draw IDs and contours'
        onlyIDs = how == 'Draw only IDs'
        nothing = how == 'Draw nothing'
        onlyCont = how == 'Draw only contours'
        only_ccaInfo = how == 'Draw only cell cycle info'
        ccaInfo_and_cont = how == 'Draw cell cycle info and contours'

        # Draw LabelItems for IDs on ax2
        y, x = obj.centroid
        t0 = time.time()
        idx = obj.label-1
        if not idx < len(self.ax2_LabelItemsIDs):
            # Create additional missing LabelItems for the current ID
            missingLen = idx-len(self.ax2_LabelItemsIDs)+1
            for i in range(missingLen):
                _IDlabel2 = pg.LabelItem()
                self.ax2_LabelItemsIDs.append(_IDlabel2)
                self.ax2.addItem(_IDlabel2)

        self.ax2_setTextID(obj, how)

        # Draw LabelItems for IDs on ax1 if requested
        if IDs_and_cont or onlyIDs or only_ccaInfo or ccaInfo_and_cont:
            # Draw LabelItems for IDs on ax2
            t0 = time.time()
            idx = obj.label-1
            if not idx < len(self.ax1_LabelItemsIDs):
                # Create addition LabelItems ax1_LabelItemsIDs the current ID
                missingLen = idx-len(self.ax1_LabelItemsIDs)+1
                for i in range(missingLen):
                    _IDlabel1 = pg.LabelItem()
                    self.ax1_LabelItemsIDs.append(_IDlabel1)
                    self.ax1.addItem(_IDlabel1)

            self.ax1_setTextID(obj, how)

        t1 = time.time()
        self.drawingLabelsTimes.append(t1-t0)

        # Draw line connecting mother and buds
        mode = self.modeComboBox.currentText()
        if mode != 'Segmentation and Tracking' and self.cca_df is not None:
            ID = obj.label
            BudMothLine = self.ax1_BudMothLines[ID-1]
            cca_df_ID = self.cca_df.loc[ID]
            ccs_ID = cca_df_ID['cell_cycle_stage']
            relationship = cca_df_ID['relationship']
            if ccs_ID == 'S' and relationship=='bud':
                emerg_frame_i = cca_df_ID['emerg_frame_i']
                if emerg_frame_i == self.frame_i:
                    pen = self.NewBudMoth_Pen
                else:
                    pen = self.OldBudMoth_Pen
                relative_ID = cca_df_ID['relative_ID']
                relative_rp_idx = self.IDs.index(relative_ID)
                relative_ID_obj = self.rp[relative_rp_idx]
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
            cont = self.get_objContours(obj)
            t1 = time.time()
            computingContoursTime = t1-t0
            self.computingContoursTimes.append(computingContoursTime)

            t0 = time.time()
            idx = ID-1
            if not idx < len(self.ax1_ContoursCurves):
                # Create addition PlotDataItems for the current ID
                missingLen = idx-len(self.ax1_ContoursCurves)+1
                for i in range(missingLen):
                    curve = pg.PlotDataItem()
                    self.ax1_ContoursCurves.append(curve)
                    self.ax1.addItem(curve)

            curveID = self.ax1_ContoursCurves[idx]
            pen = self.newIDs_cpen if ID in self.new_IDs else self.oldIDs_cpen
            curveID.setData(cont[:,0], cont[:,1], pen=pen)
            t1 = time.time()
            drawingContoursTimes = t1-t0
            self.drawingContoursTimes.append(drawingContoursTimes)


    def update_rp(self, draw=True):
        # Update rp for current self.lab (e.g. after any change)
        self.rp = skimage.measure.regionprops(self.lab)
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

        # Draw label and contours of the new IDs
        if len(newIDs)>0:
            for i, obj in enumerate(self.rp):
                ID = obj.label
                if ID in newIDs:
                    # Draw ID labels and contours of new objects
                    self.drawID_and_Contour(obj)

        # Clear contours and LabelItems of IDs that are in prev_IDs
        # but not in current IDs
        currentIDs = [obj.label for obj in self.rp]
        for prevID in prev_IDs:
            if prevID not in currentIDs:
                self.ax1_ContoursCurves[prevID-1].setData([], [])
                self.ax1_LabelItemsIDs[prevID-1].setText('')
                self.ax2_LabelItemsIDs[prevID-1].setText('')

    def updateLookuptable(self):
        lenNewLut = self.lab.max()+1
        # Build a new lut to include IDs > than original len of lut
        if lenNewLut > len(self.lut):
            numNewColors = lenNewLut-len(self.lut)
            # Index original lut
            _lut = np.zeros((lenNewLut, 3), np.uint8)
            _lut[:len(self.lut)] = self.lut
            # Pick random colors and append them at the end to recycle them
            randomIdx = np.random.randint(0,len(self.lut),size=numNewColors)
            for i, idx in enumerate(randomIdx):
                rgb = self.lut[idx]
                _lut[len(self.lut)+i] = rgb
            self.lut = _lut

        lut = self.lut[:lenNewLut].copy()
        for ID in self.binnedIDs:
            lut[ID] = lut[ID]*0.2
        for ID in self.ripIDs:
            lut[ID] = lut[ID]*0.2
        self.img2.setLookupTable(lut)

    def update_rp_metadata(self, draw=True):
        binnedIDs_xx = []
        binnedIDs_yy = []
        ripIDs_xx = []
        ripIDs_yy = []
        # Add to rp dynamic metadata (e.g. cells annotated as dead)
        for i, obj in enumerate(self.rp):
            ID = obj.label
            # IDs removed from analysis --> store info
            if ID in self.binnedIDs:
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
            if ID in self.ripIDs:
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
            self.binnedIDs_ScatterPlot.setData(binnedIDs_xx, binnedIDs_yy)
            self.ripIDs_ScatterPlot.setData(ripIDs_xx, ripIDs_yy)

    def load_fluo_data(self, fluo_path):
        print('Loading fluorescent image data...')
        # Load overlay frames and align if needed
        filename = os.path.basename(fluo_path)
        filename_noEXT, ext = os.path.splitext(filename)
        if ext == '.npy':
            fluo_data = np.load(fluo_path)
            if filename.find('aligned') != -1:
                align_ol = False
            else:
                align_ol = True
        elif ext == '.tif' or ext == '.tiff':
            align_ol = True
            fluo_data = skimage.io.imread(fluo_path)
        else:
            txt = (f'File format {ext} is not supported!\n'
                    'Choose either .tif or .npy files.')
            msg = QtGui.QMessageBox()
            msg.critical(
                self, 'File not supported', txt, msg.Ok
            )
            return None

        if align_ol:
            print('Aligning fluorescent image data...')
            images_path = self.data.images_path
            loaded_shifts, shifts_found = load.load_shifts(images_path)
            if shifts_found:
                is_3D = self.data.SizeZ > 1
                align_func = (core.align_frames_3D if is_3D
                              else core.align_frames_2D)
                aligned_frames, shifts = align_func(
                                          fluo_data,
                                          slices=None,
                                          register=False,
                                          user_shifts=loaded_shifts
                )
                aligned_filename = f'{filename_noEXT}_aligned.npy'
                aligned_path = f'{images_path}/{aligned_filename}'
                np.save(aligned_path, aligned_frames, allow_pickle=False)
                fluo_data = aligned_frames
            else:
                align_path = f'{images_path}/..._align_shift.npy'
                txt = (f'File containing alignment shifts not found!'
                       f'Looking for:\n\n"{align_path}"\n\n'
                       'Overlay images cannot be aligned to the cells image.\n'
                       'Run the segmentation script again to align the cells image.')
                msg = QtGui.QMessageBox()
                msg.critical(
                    self, 'Shifts file not found!', txt, msg.Ok
                )
                return None

        return fluo_data

    def overlay_cb(self):
        if self.overlayButton.isChecked():
            prompt = True

            # Check if there is already loaded data
            if self.data.fluo_data_dict:
                items = self.data.fluo_data_dict.keys()
                selectFluo = QDialogListbox(
                    'Select fluorescent image(s) to overlay',
                    'Select fluorescent image(s) to overlay\n'
                    'You can select one or more images',
                    items
                )
                selectFluo.show()
                if selectFluo.cancel:
                    prompt = True
                else:
                    prompt = False
                    for key in selectFluo.selectedItemsText:
                        ol_data = self.data.fluo_data_dict[key]

            if prompt:
                ol_path = prompts.file_dialog(
                                      title='Select image file to overlay',
                                      initialdir=self.data.images_path)
                if not ol_path:
                    return

                ol_data = self.load_fluo(ol_path)

            self.data.ol_frames = ol_data

            self.colorButton.setColor((255,255,0))

            cells_img = self.data.img_data[self.frame_i]
            fluo_img = self.data.ol_frames[self.frame_i]
            img = self.get_overlay(fluo_img, cells_img)
            self.img1.setImage(img)

            print('Done.')
            self.alphaScrollBar.setDisabled(False)
            self.colorButton.setDisabled(False)

        else:
            img = self.data.img_data[self.frame_i]
            self.img1.setImage(img)

    def get_overlay(self, img, ol_img, ol_brightness=4):
        ol_RGB_val = [v/255 for v in self.colorButton.color().getRgb()[:3]]
        ol_alpha = self.alphaScrollBar.value()/20
        img_rgb = gray2rgb(img_as_float(img))*ol_RGB_val
        ol_img_rgb = gray2rgb(img_as_float(ol_img))
        overlay = (ol_img_rgb*(1.0 - ol_alpha)+img_rgb*ol_alpha)*ol_brightness
        overlay = overlay/overlay.max()
        overlay = (np.clip(overlay, 0, 1)*255).astype(np.uint8)
        return overlay

    def updateOverlay(self, button):
        cells_img = self.data.img_data[self.frame_i]
        fluo_img = self.data.ol_frames[self.frame_i]
        img = self.get_overlay(fluo_img, cells_img)
        self.img1.setImage(img)

    def updateALLimg(self, image=None, never_visited=True):
        self.frameLabel.setText(
                 f'Current frame = {self.frame_i+1}/{self.num_segm_frames}')

        if image is None:
            cells_img = self.data.img_data[self.frame_i].copy()
        else:
            cells_img = image

        if self.overlayButton.isChecked():
            cells_img = self.data.img_data[self.frame_i]
            fluo_img = self.data.ol_frames[self.frame_i]
            img = self.get_overlay(fluo_img, cells_img)
        else:
            img = cells_img

        lab = self.lab

        self.img1.setImage(img)
        self.img2.setImage(lab)
        self.updateLookuptable()

        self.clear_prevItems()

        # # Red or green border depending if visited or not
        # pen = self.RedLinePen if never_visited else self.GreenLinePen
        # Y, X = img.shape
        # off = 2 # rect offset
        # xxRect = [-off, -off, X+off, X+off, -off]
        # yyRect = [-off, Y+off, Y+off, -off, -off]
        # self.ax1BorderLine.setData(xxRect, yyRect, pen=pen)
        # self.ax2BorderLine.setData(xxRect, yyRect, pen=pen)

        self.computingContoursTimes = []
        self.drawingLabelsTimes = []
        self.drawingContoursTimes = []
        # Annotate cell ID and draw contours
        for i, obj in enumerate(self.rp):
            self.drawID_and_Contour(obj)

        # print('------------------------------------')
        # print(f'Drawing labels = {np.sum(self.drawingLabelsTimes):.3f} s')
        # print(f'Computing contours = {np.sum(self.computingContoursTimes):.3f} s')
        # print(f'Drawing contours = {np.sum(self.drawingContoursTimes):.3f} s')

        # Update annotated IDs (e.g. dead cells)
        self.update_rp_metadata()

        self.brushID = lab.max()+1

    def checkIDs_LostNew(self):
        if self.frame_i == 0:
            return
        prev_rp = self.allData_li[self.frame_i-1]['regionprops']
        prev_IDs = [obj.label for obj in prev_rp]
        curr_IDs = [obj.label for obj in self.rp]
        lost_IDs = [ID for ID in prev_IDs if ID not in curr_IDs]
        new_IDs = [ID for ID in curr_IDs if ID not in prev_IDs]
        self.old_IDs = prev_IDs
        self.IDs = curr_IDs
        warn_txt = ''
        htmlTxt = ''
        if lost_IDs:
            warn_txt = f'Cells IDs lost in current frame: {lost_IDs}'
            color = 'red'
            htmlTxt = (
                f'<font color="red">{warn_txt}</font>'
            )
        if new_IDs:
            warn_txt = f'New cells IDs in current frame: {new_IDs}'
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
        self.lost_IDs = lost_IDs
        self.new_IDs = new_IDs
        self.titleLabel.setText(htmlTxt)


    def tracking(self, onlyIDs=[], enforce=False, DoManualEdit=True,
                 storeUndo=False, prev_lab=None, prev_rp=None,
                 return_lab=False):
        if self.frame_i == 0:
            return

        # Disable tracking for already visited frames
        if self.allData_li[self.frame_i]['labels'] is not None:
            self.disableTrackingCheckBox.setChecked(True)
        else:
            self.disableTrackingCheckBox.setChecked(False)

        """
        Track only frames that were NEVER visited or the user
        specifically requested to track:
            - Never visited --> NOT self.disableTrackingCheckBox.isChecked()
            - User requested --> self.UserEnforced_Tracking
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
            print('-------------')
            print(f'Frame {self.frame_i+1} NOT tracked')
            print('-------------')
            self.checkIDs_LostNew()
            return

        print('-------------')
        print(f'Frame {self.frame_i+1} tracked')
        print('-------------')
        self.disableTrackingCheckBox.setChecked(False)


        if storeUndo:
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)

        if prev_lab is None:
            prev_lab = self.allData_li[self.frame_i-1]['labels']
        if prev_rp is None:
            prev_rp = self.allData_li[self.frame_i-1]['regionprops']
        IDs_prev = []
        IDs_curr_untracked = [obj.label for obj in self.rp]
        IoA_matrix = np.zeros((len(self.rp), len(prev_rp)))

        # For each ID in previous frame get IoA with all current IDs
        # Each
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
        tracked_lab = self.lab
        new_tracked_IDs_2 = []
        if new_untracked_IDs:
            # Relabel new untracked IDs with big number to make sure they are unique
            max_ID = max(IDs_curr_untracked)
            new_tracked_IDs = [max_ID*(i+2) for i in range(len(new_untracked_IDs))]
            tracked_lab = self.np_replace_values(tracked_lab, new_untracked_IDs,
                                                 new_tracked_IDs)
        if tracked_IDs:
            # Relabel old IDs with respective tracked IDs
            tracked_lab = self.np_replace_values(tracked_lab, old_IDs, tracked_IDs)
        if new_untracked_IDs:
            # Relabel new untracked IDs sequentially
            max_ID = max(IDs_prev)
            new_tracked_IDs_2 = [max_ID+i+1 for i in range(len(new_untracked_IDs))]
            tracked_lab = self.np_replace_values(tracked_lab, new_tracked_IDs,
                                                 new_tracked_IDs_2)

        if DoManualEdit:
            # Correct tracking with manually changed IDs
            rp = skimage.measure.regionprops(tracked_lab)
            IDs = [obj.label for obj in rp]
            self.ManuallyEditTracking(tracked_lab, IDs)

        # Update labels, regionprops and determine new and lost IDs
        self.lab = tracked_lab
        self.update_rp()
        self.checkIDs_LostNew()

    def ManuallyEditTracking(self, tracked_lab, allIDs):
        # Correct tracking with manually changed IDs
        for y, x, new_ID in self.editID_info:
            old_ID = tracked_lab[y, x]
            if new_ID in allIDs:
                tempID = tracked_lab.max() + 1
                tracked_lab[tracked_lab == old_ID] = tempID
                tracked_lab[tracked_lab == new_ID] = old_ID
                tracked_lab[tracked_lab == tempID] = new_ID
            else:
                tracked_lab[tracked_lab == old_ID] = new_ID

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


    def undo_changes_future_frames(self):
        for i in range(self.frame_i, self.num_frames):
            if self.allData_li[i]['labels'] is None:
                break

            self.allData_li[i] = {
                                     'regionprops': [],
                                     'labels': None,
                                     'segm_metadata_df': None
             }

    # Slots
    def newFile(self):
        pass

    def openFile(self, checked=False, exp_path=None):
        self.modeComboBox.setCurrentIndex(0)

        if self.slideshowWin is not None:
            self.slideshowWin.close()

        if exp_path is None:
            exp_path = prompts.folder_dialog(
                title='Select experiment folder containing Position_n folders'
                      'or specific Position_n folder')

        if os.path.basename(exp_path).find('Position_') != -1:
            is_pos_folder = True
        else:
            is_pos_folder = False

        if os.path.basename(exp_path).find('Images') != -1:
            is_images_folder = True
        else:
            is_images_folder = False

        self.titleLabel.setText('Loading data...', color='w')

        if exp_path == '':
            self.titleLabel.setText(
                'File --> Open or Open recent to start the process',
                color='w')
            return

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
                return

            select_folder.run_widget(values, allow_abort=False)
            pos_foldername = select_folder.selected_pos[0]
            images_path = f'{exp_path}/{pos_foldername}/Images'

            if select_folder.was_aborted:
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process',
                    color='w')
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
            ch_name_selector.prompt(ch_names)
            if ch_name_selector.was_aborted:
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process',
                    color='w')
                return
            if warn:
                user_ch_name = prompts.single_entry_messagebox(
                    title='Channel name not found',
                    entry_label=ch_name_not_found_msg,
                    input_txt=ch_name_selector.channel_name,
                    toplevel=False
                ).entry_txt
            else:
                user_ch_name = ch_name_selector.channel_name

        img_aligned_found = False
        for filename in os.listdir(images_path):
            if filename.find(f'_phc_aligned.npy') != -1:
                img_path = f'{images_path}/{filename}'
                new_filename = filename.replace('phc_aligned.npy',
                                                f'{user_ch_name}_aligned.npy')
                dst = f'{images_path}/{new_filename}'
                os.rename(img_path, dst)
                filename = new_filename
            if filename.find(f'{user_ch_name}_aligned.npy') != -1:
                img_path = f'{images_path}/{filename}'
                img_aligned_found = True
        if not img_aligned_found:
            err_msg = ('Aligned frames file not found. '
                       'You need to run the segmentation script first.')
            self.titleLabel.setText(err_msg, color='r')
            raise FileNotFoundError(err_msg)
        print(f'Loading {img_path}...')

        self.init_frames_data(img_path, user_ch_name)

        # Ask whether to load fluorescent images
        msg = QtGui.QMessageBox()
        load_fluo = msg.question(
            self, 'Load fluorescent images?',
            'Do you also want to load fluorescent images? You can load as '
            'many channels as you want.\n\n'
            'If you load fluorescent images then the software will calcualte '
            'metrics for each loaded fluorescent channel such as min, max, mean, '
            'quantiles, etc. for each segmented object.\n\n'
            'NOTE: You can always load them later with '
            'File --> Load fluorescent images',
            msg.Yes | msg.No
        )
        if load_fluo == msg.Yes:
            fluo_paths = prompts.multi_files_dialog(
                title='Select one or multiple fluorescent images')

            for fluo_path in fluo_paths:
                filename = os.path.basename(fluo_path)
                fluo_data = self.load_fluo_data(fluo_path)
                self.data.fluo_data_dict[filename] = fluo_data

        # Connect events at the end of loading data process
        self.gui_connectGraphicsEvents()
        self.gui_connectEditActions()

        self.titleLabel.setText(
            'Data successfully loaded. Right/Left arrow to navigate frames',
            color='w')

        self.addToRecentPaths(exp_path)

    def addToRecentPaths(self, exp_path):
        src_path = os.path.dirname(os.path.realpath(__file__))
        recentPaths_path = os.path.join(
            src_path, 'temp', 'recentPaths.csv'
        )
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            recentPaths = df['path'].to_list()
            if exp_path not in recentPaths:
                recentPaths.insert(0, exp_path)
            # Keep max 20 recent paths
            if len(recentPaths) > 20:
                recentPaths.pop(-1)
        else:
            recentPaths = [exp_path]
        df = pd.DataFrame({'path': recentPaths})
        df.index.name = 'index'
        df.to_csv(recentPaths_path)


    def showInExplorer(self):
        systems = {
            'nt': os.startfile,
            'posix': lambda foldername: os.system('xdg-open "%s"' % foldername),
            'os2': lambda foldername: os.system('open "%s"' % foldername)
             }

        systems.get(os.name, os.startfile)(self.images_path)

    def add_static_metadata_df(self, df, rp):
        # ADD HERE FLUORESCENT METRICS and segmentation ...
        # Add metrics that can be calculated at the end of the process
        # such as cell volume, cell area etc.
        zyx_vox_dim =self.data.zyx_vox_dim

        # Calc volume
        vox_to_fl = zyx_vox_dim[1]*(zyx_vox_dim[2]**2)
        yx_pxl_to_um2 = zyx_vox_dim[1]*zyx_vox_dim[2]
        IDs_vol_vox = [0]*len(rp)
        IDs_area_pxl = [0]*len(rp)
        IDs_vol_fl = [0]*len(rp)
        IDs_area_um2 = [0]*len(rp)
        for i, obj in enumerate(rp):
            rotate_ID_img = skimage.transform.rotate(
                obj.image.astype(np.uint8), -(obj.orientation*180/np.pi),
                resize=True, order=3, preserve_range=True
            )
            radii = np.sum(rotate_ID_img, axis=1)/2
            vol_vox = np.sum(np.pi*(radii**2))
            IDs_vol_vox[i] = vol_vox
            IDs_area_pxl[i] = obj.area
            IDs_vol_fl[i] = vol_vox*vox_to_fl
            IDs_area_um2[i] = obj.area*yx_pxl_to_um2

        df['cell_area_pxl'] = IDs_area_pxl
        df['cell_vol_vox'] = IDs_vol_vox
        df['cell_area_um2'] = IDs_area_um2
        df['cell_vol_fl'] = IDs_vol_fl


    def saveFile(self):
        self.app.setOverrideCursor(Qt.WaitCursor)
        try:
            segm_npy_path = self.data.segm_npy_path
            acdc_output_csv_path = self.data.acdc_output_csv_path
            last_tracked_i_path = self.data.last_tracked_i_path
            segm_npy = np.copy(self.data.segm_data)
            segm_npy[self.frame_i] = self.lab
            segm_metadata_df_li = [None]*self.num_frames

            # Create list of dataframes from segm_metadata_df on HDD
            if self.data.segm_metadata_df is not None:
                for frame_i, df in self.data.segm_metadata_df.groupby(level=0):
                    segm_metadata_df_li[frame_i] = df.loc[frame_i]

            for frame_i, data_dict in enumerate(self.allData_li):
                # Build segm_npy
                lab = data_dict['labels']
                if lab is not None:
                    segm_npy[frame_i] = lab
                else:
                    break

                segm_metadata_df = data_dict['segm_metadata_df']

                # Build segm_metadata_df and index it in each frame_i
                if segm_metadata_df is not None:
                    rp = data_dict['regionprops']
                    self.add_static_metadata_df(segm_metadata_df, rp)
                    segm_metadata_df_li[frame_i] = segm_metadata_df

            # Remove None and concat dataframe
            keys = []
            df_li = []
            for i, df in enumerate(segm_metadata_df_li):
                if df is not None:
                    df_li.append(df)
                    keys.append(i)

            try:
                all_frames_metadata_df = pd.concat(
                    df_li, keys=keys, names=['frame_i', 'Cell_ID']
                )

                # Save segmentation metadata
                all_frames_metadata_df.to_csv(acdc_output_csv_path)
                self.data.segm_metadata_df = all_frames_metadata_df
            except:
                traceback.print_exc()
                pass

            # Save segmentation file
            np.save(segm_npy_path, segm_npy)
            self.data.segm_data = segm_npy

            # Save last tracked frame
            with open(last_tracked_i_path, 'w+') as txt:
                txt.write(str(frame_i))

            print('--------------')
            print(f'Saved data until frame number {frame_i+1}')
            print('--------------')
        except:
            traceback.print_exc()
        finally:
            self.app.restoreOverrideCursor()




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
        # Step 0. Remove the old options from the menu
        self.openRecentMenu.clear()
        # Step 1. Read recent Paths
        src_path = os.path.dirname(os.path.realpath(__file__))
        recentPaths_path = os.path.join(
            src_path, 'temp', 'recentPaths.csv'
        )
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
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

    def openRecentFile(self, path):
        print(f'Opening recent folder: {path}')
        self.openFile(exp_path=path)

    def closeEvent(self, event):
        if self.slideshowWin is not None:
            self.slideshowWin.close()
        if self.editToolBar.isEnabled():
            msg = QtGui.QMessageBox()
            msg.closeEvent = self.saveMsgCloseEvent
            save = msg.question(
                self, 'Save?', 'Do you want to save segmentation data?',
                msg.Yes | msg.No | msg.Cancel
            )
            if save == msg.Yes:
                self.saveFile()
                event.accept()
            elif save == msg.No:
                event.accept()
            else:
                event.ignore()

    def saveMsgCloseEvent(self, event):
        print('closed')

if __name__ == "__main__":
    # Create the application
    app = QApplication(sys.argv)
    # Apply dark mode
    file = QFile(":/dark.qss")
    file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(file)
    # app.setStyleShefet(stream.readAll())
    # Create and show the main window
    win = Yeast_ACDC_GUI(app)
    win.show()
    # Apply style
    app.setStyle(QtGui.QStyleFactory.create('Fusion'))
    # Run the event loop
    sys.exit(app.exec_())
