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
import traceback
import time
from functools import partial

import cv2
import numpy as np
import pandas as pd
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

        # Contour pen
        self.cpen = pg.mkPen(color='r', width=2)

        # New bud-mother line pen
        self.NewBudMoth_Pen = pg.mkPen(color='r', width=3, style=Qt.DashLine)

        # Old bud-mother line pen
        self.OldBudMoth_Pen = pg.mkPen(color=(255,165,0), width=2,
                                       style=Qt.DashLine)

        # Create enough PlotDataItems and LabelItems to draw contours and IDs
        numItems = maxID+10
        self.ax1_ContoursCurves = []
        self.ax1_BudMothLines = []
        self.ax1_LabelItemsIDs = []
        self.ax2_LabelItemsIDs = []
        for i in range(numItems):
            # Contours on ax1
            ContCurve = pg.PlotDataItem(pen=self.cpen)
            self.ax1_ContoursCurves.append(ContCurve)
            self.ax1.addItem(ContCurve)

            # Contours on ax1
            BudMothLine = pg.PlotDataItem()
            self.ax1_BudMothLines.append(BudMothLine)
            self.ax1.addItem(BudMothLine)

            # LabelItems on ax1
            ax1_IDlabel = pg.LabelItem(
                    text='',
                    color='FA0000',
                    bold=True,
                    size='10pt'
            )
            self.ax1_LabelItemsIDs.append(ax1_IDlabel)
            self.ax1.addItem(ax1_IDlabel)

            # LabelItems on ax2
            ax2_IDlabel = pg.LabelItem(
                    text='',
                    color='FA0000',
                    bold=True,
                    size='10pt'
            )
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
                self.storeUndoRedoStates()
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
                self.storeUndoRedoStates()
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
        elif mid_click and mode == 'Segmentation and tracking':
            # Store undo state before modifying stuff
            self.storeUndoRedoStates()
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            delID = self.lab[ydata, xdata]
            self.lab[self.lab==delID] = 0

            # Update data (rp, etc)
            prev_IDs = [obj.label for obj in self.rp]
            self.update_rp()

            self.img2.updateImage()

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
                # self.separateBudButton.setChecked(False)
                return
            # Store undo state before modifying stuff
            self.storeUndoRedoStates()
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
                # self.mergeIDsButton.setChecked(False)
                return
            # Store undo state before modifying stuff
            self.storeUndoRedoStates()
            self.firstID = ID

        # Edit ID
        elif right_click and self.editID_Button.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            ID = self.lab[ydata, xdata]
            if ID == 0:
                # self.editID_Button.setChecked(False)
                return

            self.disableAutoActivateViewerWindow = True
            prev_IDs = [obj.label for obj in self.rp]
            editID = apps.editID_QWidget(ID, prev_IDs)
            editID.exec_()
            if editID.cancel:
                self.disableAutoActivateViewerWindow = False
                self.editID_Button.setChecked(False)
                return

            # Store undo state before modifying stuff
            self.storeUndoRedoStates()
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
                    y, x = int(round(x)), int(round(y))
                    self.editID_info.append((y, x, new_ID))
                    obj = self.rp[new_ID_idx]
                    y, x = obj.centroid
                    y, x = int(round(x)), int(round(y))
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
                    y, x = int(round(x)), int(round(y))
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

        # Annotate cell as removed from the analysis
        elif right_click and self.binCellButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            ID = self.lab[ydata, xdata]
            if ID == 0:
                # self.binCellButton.setChecked(False)
                return

            # Store undo state before modifying stuff
            self.storeUndoRedoStates()

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
                # self.ripCellButton.setChecked(False)
                return

            # Store undo state before modifying stuff
            self.storeUndoRedoStates()

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

            # Update data (rp, etc)
            prev_IDs = [obj.label for obj in self.rp]
            self.update_rp()

            self.update_IDsContours(
                prev_IDs, newIDs=erasedIDs
            )

        # Brush mouse release --> update IDs and contours
        elif self.isMouseDragImg2 and self.brushButton.isChecked():
            self.isMouseDragImg2 = False

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
                # self.mergeIDsButton.setChecked(False)
                return
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
                return

            relationship = self.cca_df.at[ID, 'relationship']
            ccs = self.cca_df.at[ID, 'cell_cycle_stage']

            if relationship == 'bud':
                txt = (f'You clicked on ID {ID} which is a BUD.\n'
                       'To assign a bud to a cell start by clicking on a bud '
                       'and release on a cell in G1')
                msg = QtGui.QMessageBox()
                msg.critical(
                    self, 'Released on a bud', txt, msg.Ok
                )
                return

            elif ccs != 'G1':
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

    def gui_mousePressEventImg1(self, event):
        mode = str(self.modeComboBox.currentText())
        is_cca_on = mode == 'Cell cycle analysis'
        right_click = event.button() == Qt.MouseButton.RightButton
        left_click = event.button() == Qt.MouseButton.LeftButton

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
            self.storeUndoRedoStates()
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            lab = self.lab
            Y, X = lab.shape
            brushSize = self.brushSizeSpinbox.value()
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
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
                self.tracking()
                newIDs = [self.lab[ymin:ymax, xmin:xmax][mask][0]]

                # Update colors to include a new color for the new ID
                self.img2.setImage(self.lab)
                self.updateLookuptable()

                # Update contours
                self.update_IDsContours(prev_IDs, newIDs=newIDs)

                # Update brush ID. Take care of disappearing cells to remember
                # to not use their IDs anymore in the future
                if self.lab.max()+1 > self.brushID:
                    self.brushID = self.lab.max()+1
                else:
                    self.brushID += 1

        elif right_click and mode == 'Segmentation and tracking':
            # Allow right-click actions on both images
            self.gui_mousePressEventImg2(event)

        elif right_click and is_cca_on and not self.assignMothBudButton.isChecked():
            if self.cca_df is None:
                return

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            ID = self.lab[ydata, xdata]
            if ID == 0:
                return

            # Annotate or undo division
            self.manualCellCycleAnnotation(ID)

        elif right_click and self.assignMothBudButton.isChecked():
            self.clickedOnBud = False
            if self.cca_df is None:
                return

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            ID = self.lab[ydata, xdata]
            if ID == 0:
                return

            relationship = self.cca_df.at[ID, 'relationship']
            if relationship != 'bud':
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

    def annotateDivision(self, division_i ,cca_df, ID, relID, ccs, ccs_relID):
        # Correct as follows:
        # If S then correct to G1 and +1 on generation number
        store = False
        if ccs == 'S':
            cca_df.at[ID, 'cell_cycle_stage'] = 'G1'
            cca_df.at[ID, 'generation_num'] += 1
            cca_df.at[ID, 'division_frame_i'] = division_i
            store = True
        if ccs_relID == 'S':
            cca_df.at[relID, 'cell_cycle_stage'] = 'G1'
            cca_df.at[relID, 'generation_num'] += 1
            cca_df.at[relID, 'division_frame_i'] = division_i
            store = True
        return store

    def undoDivisionAnnotation(self, cca_df, ID, relID, ccs, ccs_relID):
        # Correct as follows:
        # If G1 then correct to S and -1 on generation number
        store = False
        if ccs == 'G1':
            cca_df.at[ID, 'cell_cycle_stage'] = 'S'
            cca_df.at[ID, 'generation_num'] -= 1
            cca_df.at[ID, 'division_frame_i'] = -1
            store = True
        if ccs_relID == 'G1':
            cca_df.at[relID, 'cell_cycle_stage'] = 'S'
            cca_df.at[relID, 'generation_num'] -= 1
            cca_df.at[relID, 'division_frame_i'] = -1
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
        In this case we assign all those frames to G1 and +1 generation number

        If we undo the annotation (right click on a cell in G1) then it will
        correct both past and future annotated frames (if present).
        Frames to correct are those frames where both the mother and the bud
        are annotated as G1 phase cells.
        In this case we assign all those frames to G1 and -1 generation number
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
        if self.isAltDown:
            self.app.setOverrideCursor(Qt.SizeAllCursor)
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
                self.app.setOverrideCursor(Qt.ArrowCursor)
                self.wcLabel.setText(f'')
        except:
            self.app.setOverrideCursor(Qt.ArrowCursor)
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

    def gui_hoverEventImg2(self, event):
        if self.isAltDown:
            self.app.setOverrideCursor(Qt.SizeAllCursor)
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
                self.app.setOverrideCursor(Qt.ArrowCursor)
                self.wcLabel.setText(f'')
        except:
            self.app.setOverrideCursor(Qt.ArrowCursor)
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
        self.disableTrackingCheckBox.toggled.connect(self.disableTracking)
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
        self.storeUndoRedoStates()
        self.lab = np.load(self.data.segm_npy_path)[self.frame_i]
        self.update_rp()
        self.updateALLimg()
        self.app.setOverrideCursor(Qt.ArrowCursor)

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
                self.navigateToolBar.setEnabled(True)
                self.modeToolBar.setEnabled(True)
                self.disableTrackingCheckBox.setChecked(True)
                self.assignMothBudButton.setDisabled(False)
                self.drawIDsContComboBox.clear()
                self.drawIDsContComboBox.addItems(
                                        self.drawIDsContComboBoxCcaItems)

        elif mode == 'Viewer':
            self.setEnabledToolbarButton(enabled=False)
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
            # self.app.setOverrideCursor(Qt.ArrowCursor)
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
            # self.app.setOverrideCursor(self.brushCursor)

    def equalizeHist(self):
        # Store undo state before modifying stuff
        self.storeUndoRedoStates()
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
            # self.app.setOverrideCursor(Qt.ArrowCursor)
        else:
            self.enableSizeSpinbox(True)
            # self.app.setOverrideCursor(self.brushCursor)

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
        # elif ev.key() == Qt.Key_Left:
        #     self.prev_cb()
        # elif ev.text() == 'b':
        #     self.BrushEraser_cb(ev)

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key_Alt:
            self.app.setOverrideCursor(Qt.ArrowCursor)
            self.isAltDown = False

    def setUncheckedAllButtons(self):
        for button in self.checkableButtons:
            button.setChecked(False)

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

    def storeUndoRedoStates(self):
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

    def disableTracking(self):
        pass

    def repeatTracking(self):
        self.tracking(enforce=True, DoManualEdit=False)
        if self.editID_info:
            editIDinfo = [
                f'Replace ID {self.lab[y,x]} with {newID}'
                for y, x, newID in self.editID_info
            ]
            msg = QtGui.QMessageBox()
            msg.setIcon(msg.Information)
            msg.setText("You required to repeat tracking but there are "
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
        if self.frame_i <= 0 and mode == 'Cell cycle analysis':
            IDs = [obj.label for obj in self.rp]
            init_cca_df_frame0 = apps.cca_df_frame0(IDs, self.cca_df)
            if init_cca_df_frame0.cancel:
                return
            self.cca_df = init_cca_df_frame0.df
        self.app.setOverrideCursor(Qt.WaitCursor)
        if self.frame_i < self.num_segm_frames-1:
            # Store data for current frame
            self.store_data(debug=False)
            # Go to next frame
            self.frame_i += 1
            proceed = self.get_data()
            if not proceed:
                self.frame_i -= 1
                return
            self.tracking()
            self.auto_cca_df()
            self.updateALLimg()
        else:
            msg = 'You reached the last segmented frame!'
            print(msg)
            self.titleLabel.setText(msg)
        if self.slideshowWin is not None:
            self.slideshowWin.frame_i = self.frame_i
            self.slideshowWin.update_img()
        self.app.setOverrideCursor(Qt.ArrowCursor)

    def prev_cb(self):
        self.app.setOverrideCursor(Qt.WaitCursor)
        if self.frame_i > 0:
            self.store_data()
            self.frame_i -= 1
            self.get_data()
            self.tracking()
            self.updateALLimg()
        else:
            msg = 'You reached the first frame!'
            print(msg)
            self.titleLabel.setText(msg)
        if self.slideshowWin is not None:
            self.slideshowWin.frame_i = self.frame_i
            self.slideshowWin.update_img()
        self.app.setOverrideCursor(Qt.ArrowCursor)

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
        self.disableAutoActivateViewerWindow = False
        self.enforceSeparation = False
        self.isAltDown = False
        self.manual_newID_coords = []
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
        if self.data.last_tracked_i is not None:
            self.app.setOverrideCursor(Qt.ArrowCursor)
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
                self.updateALLimg()
        else:
            self.frame_i = 0
            self.get_data()
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
        for i, obj in enumerate(self.rp):
            is_cell_dead_li[i] = obj.dead
            is_cell_excluded_li[i] = obj.excluded
            IDs[i] = obj.label
            xx_centroid[i] = obj.centroid[1]
            yy_centroid[i] = obj.centroid[0]

        self.allData_li[self.frame_i]['segm_metadata_df'] = pd.DataFrame(
            {
                        'Cell_ID': IDs,
                        'is_cell_dead': is_cell_dead_li,
                        'is_cell_excluded': is_cell_excluded_li,
                        'x_centroid': xx_centroid,
                        'y_centroid': yy_centroid
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
        return point, nearest_point

    def auto_cca_df(self):
        """
        Mother and bud IDs are determined as the two objects (old and new) that
        have the smallest euclidean distance between all pairs of new-old points
        of the new and old objects' contours. Note that mother has to be in G1
        """
        # If cca_df for current frame is None it means we never annotated it
        # and we automatically infer annotation
        if self.cca_df is None:
            df = self.allData_li[self.frame_i-1]['segm_metadata_df']
            if df is not None:
                if 'cell_cycle_stage' in df.columns:
                    self.cca_df = df[['cell_cycle_stage',
                                      'generation_num',
                                      'relative_ID',
                                      'relationship',
                                      'emerg_frame_i',
                                      'division_frame_i']].copy()

                    # If there are no new IDs we are done
                    if not self.new_IDs:
                        return

                    # Calculate contour of all old IDs in G1 (potential mothers)
                    oldIDs_contours = []
                    for obj in self.rp:
                        ID = obj.label
                        if ID in self.old_IDs:
                            ccs = self.cca_df.at[ID, 'cell_cycle_stage']
                            if ccs == 'G1':
                                cont = self.get_objContours(obj)
                                oldIDs_contours.append(cont)
                    oldIDs_contours = np.concatenate(oldIDs_contours, axis=0)

                    # For each new ID calculate nearest old ID contour
                    for obj in self.rp:
                        ID = obj.label
                        if ID in self.new_IDs:
                            new_ID_cont = self.get_objContours(obj)
                            _, nearest_xy = self.nearest_point_2Dyx(
                                                new_ID_cont, oldIDs_contours)
                            mothID = self.lab[nearest_xy[1], nearest_xy[0]]
                            self.cca_df.at[mothID, 'relative_ID'] = ID
                            self.cca_df.at[mothID, 'cell_cycle_stage'] = 'S'

                            self.cca_df.loc[ID] = {
                                'cell_cycle_stage': 'S',
                                'generation_num': 0,
                                'relative_ID': mothID,
                                'relationship': 'bud',
                                'emerg_frame_i': self.frame_i,
                                'division_frame_i': -1
                            }

                    self.store_cca_df()


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
        proceed = True
        if self.frame_i > 2:
            # Remove undo states from 2 frames back to avoid memory issues
            self.UndoRedoStates[self.frame_i-2] = []
            # Check if current frame contains undo states (not empty list)
            if self.UndoRedoStates[self.frame_i]:
                self.undoAction.setDisabled(False)
            else:
                self.undoAction.setDisabled(True)
        self.UndoCount = 0
        self.editID_info = []
        # If stored labels is None then it is the first time we visit this frame
        if self.allData_li[self.frame_i]['labels'] is None:
            if str(self.modeComboBox.currentText()) == 'Cell cycle analysis':
                # Warn that we are visiting a frame that was never segm-checked
                # on cell cycle analysis mode
                msg = QtGui.QMessageBox()
                warn_cca = msg.critical(
                    self, 'Never checked segmentation on requested frame',
                    'Segmentation and tracking was never checked from '
                    f'frame {self.frame_i+1} onward.\n To ensure correct cell '
                    'cell cycle analysis we recommend to first visit frames '
                    f'{self.frame_i+1}-end with "Segmentation and tracking mode."'
                    'However you can decide to continue with cell cycle analysis '
                    'anyway\n\n.'
                    'Do you want to proceed?',
                    msg.Ok
                )
                return proceed
            # Requested frame was never visited before. Load from HDD
            self.lab = self.data.segm_data[self.frame_i].copy()
            self.rp = skimage.measure.regionprops(self.lab)
            if self.data.segm_metadata_df is not None:
                frames = self.data.segm_metadata_df.index.get_level_values(0)
                if self.frame_i in frames:
                    # Since there was already segmentation metadata from
                    # previous analysis add it to current metadata
                    df = self.data.segm_metadata_df.loc[self.frame_i]
                    binnedIDs_df = df[df['is_cell_excluded']]
                    binnedIDs = set(binnedIDs_df.index).union(self.binnedIDs)
                    self.binnedIDs = binnedIDs
                    ripIDs_df = df[df['is_cell_dead']]
                    ripIDs = set(ripIDs_df.index).union(self.ripIDs)
                    self.ripIDs = ripIDs
            self.get_cca_df()
        else:
            # Requested frame was already visited. Load from RAM.
            self.lab = self.allData_li[self.frame_i]['labels'].copy()
            self.rp = skimage.measure.regionprops(self.lab)
            df = self.allData_li[self.frame_i]['segm_metadata_df']
            binnedIDs_df = df[df['is_cell_excluded']]
            self.binnedIDs = set(binnedIDs_df.index)
            ripIDs_df = df[df['is_cell_dead']]
            self.ripIDs = set(ripIDs_df.index)
            self.get_cca_df()

        return proceed

    def init_cca(self):
        if self.data.last_tracked_i is None:
            txt = (
                'On this dataset you never checked that the segmentation '
                'and tracking are correct.\n'
                'You first have to check (and eventually correct) some frames'
                'in "Segmentation and tracking" mode before proceeding '
                'with cell cycle analysis.')
            msg = QtGui.QMessageBox()
            msg.critical(
                self, 'Tracking check not performed', txt, msg.Ok
            )
            return

        proceed = True
        if self.cca_df is None:
            last_cca_frame_i = 0
        else:
            last_cca_frame_i = 0
            # Determine last annotated frame index
            for i, dict_frame_i in enumerate(self.allData_li):
                df = dict_frame_i['segm_metadata_df']
                if df is not None:
                    if 'cell_cycle_stage' not in df.columns:
                        last_cca_frame_i = i-1
                        break

        if self.frame_i != last_cca_frame_i:
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
        else:
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

            cca_colNames = ['cell_cycle_stage', 'generation_num',
                            'relative_ID', 'relationship',
                            'emerg_frame_i', 'division_frame_i']
            df.drop(cca_colNames, axis=1, inplace=True)





    def get_cca_df(self, frame_i=None, return_df=False):
        # cca_df is None unless the metadata contains cell cycle annotations
        # NOTE: cell cycle annotations are either from the current session
        # or loaded from HDD in "init_attr" with a .question to the user
        cca_df = None
        i = self.frame_i if frame_i is None else frame_i
        df = self.allData_li[i]['segm_metadata_df']
        if df is not None:
            if 'cell_cycle_stage' in df.columns:
                cca_df = df[['cell_cycle_stage',
                             'generation_num',
                             'relative_ID',
                             'relationship',
                             'emerg_frame_i',
                             'division_frame_i']].copy()
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
                df[['cell_cycle_stage',
                    'generation_num',
                    'relative_ID',
                    'relationship',
                    'emerg_frame_i',
                    'division_frame_i']] = cca_df
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
            LabelItemID.setText(f'{ID}')
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
            txt = f'{ccs}-{generation_num}' if self.frame_i !=0 else f'{ID}'
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
                _IDlabel2 = pg.LabelItem(
                        text='',
                        color='FA0000',
                        bold=True,
                        size='10pt'
                )
                self.ax2_LabelItemsIDs.append(_IDlabel2)
                self.ax2.addItem(_IDlabel2)

        _IDlabel2 = self.ax2_LabelItemsIDs[obj.label-1]
        _IDlabel2.setText(f'{obj.label}')
        w, h = _IDlabel2.rect().right(), _IDlabel2.rect().bottom()
        _IDlabel2.setPos(x-w/2, y-h/2)

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
        if mode != 'Segmentation and tracking' and self.cca_df is not None:
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
            t0 = time.time()
            cont = self.get_objContours(obj)
            t1 = time.time()
            computingContoursTime = t1-t0
            self.computingContoursTimes.append(computingContoursTime)

            t0 = time.time()
            idx = obj.label-1
            if not idx < len(self.ax1_ContoursCurves):
                # Create addition PlotDataItems for the current ID
                missingLen = idx-len(self.ax1_ContoursCurves)+1
                for i in range(missingLen):
                    curve = pg.PlotDataItem(pen=self.cpen)
                    self.ax1_ContoursCurves.append(curve)
                    self.ax1.addItem(curve)

            curveID = self.ax1_ContoursCurves[idx]
            curveID.setData(cont[:,0], cont[:,1])
            t1 = time.time()
            drawingContoursTimes = t1-t0
            self.drawingContoursTimes.append(drawingContoursTimes)


    def update_rp(self):
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

    def overlay_cb(self):
        if self.overlayButton.isChecked():
            ol_path = prompts.file_dialog(
                                  title='Select image file to overlay',
                                  initialdir=self.data.images_path)
            if not ol_path:
                return

            self.app.setOverrideCursor(Qt.WaitCursor)
            # Load overlay frames and align if needed
            filename = os.path.basename(ol_path)
            filename_noEXT, ext = os.path.splitext(filename)

            print('Loading fluorescent image data...')
            if ext == '.npy':
                self.data.ol_frames = np.load(ol_path)
                if filename.find('aligned') != -1:
                    align_ol = False
                else:
                    align_ol = True
            elif ext == '.tif' or ext == '.tiff':
                align_ol = True
                self.data.ol_frames = skimage.io.imread(ol_path)
            else:
                self.app.setOverrideCursor(Qt.ArrowCursor)
                txt = (f'File format {ext} is not supported!\n'
                        'Choose either .tif or .npy files.')
                msg = QtGui.QMessageBox()
                msg.critical(
                    self, 'File not supported', txt, msg.Ok
                )
                return

            if align_ol:
                print('Aligning fluorescent image data...')
                images_path = self.data.images_path
                loaded_shifts, shifts_found = load.load_shifts(images_path)
                if shifts_found:
                    is_3D = self.data.SizeZ > 1
                    align_func = (core.align_frames_3D if is_3D
                                  else core.align_frames_2D)
                    aligned_frames, shifts = align_func(
                                              self.data.ol_frames,
                                              slices=None,
                                              register=False,
                                              user_shifts=loaded_shifts
                    )
                    aligned_filename = f'{filename_noEXT}_aligned.npy'
                    aligned_path = f'{images_path}/{aligned_filename}'
                    np.save(aligned_path, aligned_frames, allow_pickle=False)
                    self.data.ol_frames = aligned_frames
                else:
                    self.app.setOverrideCursor(Qt.ArrowCursor)
                    txt = ('\"..._align_shift.npy\" file not found!\n'
                           'Overlay images cannot be aligned to the cells image.')
                    msg = QtGui.QMessageBox()
                    msg.critical(
                        self, 'Shifts file not found!', txt, msg.Ok
                    )
                    return

            self.colorButton.setColor((255,255,0))

            cells_img = self.data.img_data[self.frame_i]
            fluo_img = self.data.ol_frames[self.frame_i]
            img = self.get_overlay(fluo_img, cells_img)
            self.img1.setImage(img)

            print('Done.')
            self.alphaScrollBar.setDisabled(False)
            self.colorButton.setDisabled(False)
            self.app.setOverrideCursor(Qt.ArrowCursor)

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

    def updateALLimg(self, image=None):
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

        self.computingContoursTimes = []
        self.drawingLabelsTimes = []
        self.drawingContoursTimes = []
        # Annotate cell ID and draw contours
        for i, obj in enumerate(self.rp):
            self.drawID_and_Contour(obj)

        print('------------------------------------')
        print(f'Drawing labels = {np.sum(self.drawingLabelsTimes):.3f} s')
        print(f'Computing contours = {np.sum(self.computingContoursTimes):.3f} s')
        print(f'Drawing contours = {np.sum(self.drawingContoursTimes):.3f} s')

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


    def tracking(self, onlyIDs=[], enforce=False, DoManualEdit=True):
        if self.frame_i == 0:
            return
        # Track only frames that were visited for the first time
        do_tracking = (
                (self.allData_li[self.frame_i]['labels'] is None)
                or enforce
        )
        if self.disableTrackingCheckBox.isChecked() or not do_tracking:
            self.checkIDs_LostNew()
            return
        prev_lab = self.allData_li[self.frame_i-1]['labels']
        prev_rp = self.allData_li[self.frame_i-1]['regionprops']
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

        allIDs = tracked_IDs.copy()
        tracked_IDs.extend(new_tracked_IDs_2)

        if DoManualEdit:
            # Correct tracking with manually changed IDs
            self.ManuallyEditTracking(tracked_lab, allIDs)

        # Update labels, regionprops and determine new and lost IDs
        self.lab = tracked_lab
        self.update_rp()
        self.checkIDs_LostNew()

    def ManuallyEditTracking(self, tracked_lab, allIDs):
        # Correct tracking with manually changed IDs
        for y, x, new_ID in self.editID_info:
            old_ID = tracked_lab[y, x]
            if new_ID in allIDs:
                tempID = tracked_lab + 1
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

        is_pos_folder = False
        if exp_path.find('Position_') != -1:
            is_pos_folder = True

        self.titleLabel.setText('Loading data...')

        if exp_path == '':
            self.titleLabel.setText(
                'File --> Open or Open recent to start the process')
            return

        ch_name_selector = prompts.select_channel_name(
            which_channel='segm', allow_abort=False
        )

        if not is_pos_folder:
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

            pos_foldername = select_folder.run_widget(values, allow_abort=False)

            if select_folder.was_aborted:
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process')
                return

        else:
            pos_foldername = os.path.basename(exp_path)
            exp_path = os.path.dirname(exp_path)


        images_path = f'{exp_path}/{pos_foldername}/Images'

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
                    'File --> Open or Open recent to start the process')
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
            self.titleLabel.setText(err_msg)
            raise FileNotFoundError(err_msg)

        self.app.setOverrideCursor(Qt.WaitCursor)
        print(f'Loading {img_path}...')

        self.init_frames_data(img_path, user_ch_name)

        # Connect events at the end of loading data process
        self.gui_connectGraphicsEvents()
        self.gui_connectEditActions()

        self.titleLabel.setText(
                'Data successfully loaded. Right/Left arrow to navigate frames')

        self.addToRecentPaths(exp_path)
        self.app.setOverrideCursor(Qt.ArrowCursor)

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

            self.app.setOverrideCursor(Qt.ArrowCursor)

        except:
            traceback.print_exc()
            self.app.setOverrideCursor(Qt.ForbiddenCursor)




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
    # app.setOverrideCursor(Qt.WaitCursor)
    # app.setStyleSheet(stream.readAll())
    # Create and show the main window
    win = Yeast_ACDC_GUI(app)
    win.show()
    # Apply style
    app.setStyle(QtGui.QStyleFactory.create('Fusion'))
    # Run the event loop
    sys.exit(app.exec_())
