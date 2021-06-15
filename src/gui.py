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
    QComboBox, QDial
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
        self.slideshowWin = None
        self.data_loaded = False
        self.setWindowTitle("Yeast ACDC - Segm&Track")
        self.setGeometry(0, 0, 1366, 768)

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
        if self.slideshowWin is not None and self.data_loaded:
            self.slideshowWin.setFocus(True)
            self.slideshowWin.activateWindow()

    def enterEvent(self, event):
        if self.data_loaded:
            self.setFocus(True)
            self.activateWindow()

    def gui_createImg1Widgets(self):
        # z-slice scrollbar
        self.zSlice_scrollBar_img1 = QScrollBar(Qt.Horizontal)
        self.img1_Widglayout = QtGui.QGridLayout()
        self.zSlice_scrollBar_img1.setFixedHeight(20)
        self.zSlice_scrollBar_img1.setDisabled(True)
        _z_label = QLabel('z-slice  ')
        _font = QtGui.QFont()
        _font.setPointSize(10)
        _z_label.setFont(_font)
        self.img1_Widglayout.addWidget(_z_label, 0, 0, alignment=Qt.AlignRight)
        self.img1_Widglayout.addWidget(self.zSlice_scrollBar_img1, 0, 1, 1, 10)

        # Fluorescent overlay alpha
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
            alphaScrollBar_label, 1, 0
        )
        self.img1_Widglayout.addWidget(
            alphaScrollBar, 1, 1, 1, 10
        )

        # Left, top, right, bottom
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

        # Auto image adjustment button
        proxy = QtGui.QGraphicsProxyWidget()
        equalizeHistPushButton = QPushButton("Auto")
        equalizeHistPushButton.setStyleSheet(
               'QPushButton {background-color: #282828; color: #F0F0F0;}')
        proxy.setWidget(equalizeHistPushButton)
        self.graphLayout.addItem(proxy, row=0, col=0)
        self.equalizeHistPushButton = equalizeHistPushButton

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

        # Brush Eraser circle img1
        self.brushCircle = pg.ScatterPlotItem()
        self.brushCircle.setData([], [], symbol='o', pxMode=False,
                                 brush=pg.mkBrush((255,255,255,50)),
                                 pen=pg.mkPen(width=2))
        self.plot1.addItem(self.brushCircle)

        # Eraser circle img2
        self.EraserCircle = pg.ScatterPlotItem()
        self.EraserCircle.setData([], [], symbol='o', pxMode=False,
                                 brush=None,
                                 pen=pg.mkPen(width=2, color='r'))
        self.plot2.addItem(self.EraserCircle)

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
        self.plot2.addItem(self.binnedIDs_ScatterPlot)
        self.ripIDs_ScatterPlot = pg.ScatterPlotItem()
        self.ripIDs_ScatterPlot.setData(
                                 [], [], symbol='x', pxMode=False,
                                 brush=pg.mkBrush((255,0,0,50)), size=15,
                                 pen=pg.mkPen(width=2, color='r'))
        self.plot2.addItem(self.ripIDs_ScatterPlot)


        # Title
        self.titleLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        self.titleLabel.setText('File --> Open to start the process')
        self.graphLayout.addItem(self.titleLabel, row=0, col=1, colspan=2)

        # Current frame text
        self.frameLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        self.frameLabel.setText(' ')
        self.graphLayout.addItem(self.frameLabel, row=2, col=0, colspan=3)

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
        left_click = event.button() == Qt.MouseButton.LeftButton
        mid_click = event.button() == Qt.MouseButton.MidButton
        right_click = event.button() == Qt.MouseButton.RightButton
        # Erase with brush and left click on the right image
        # NOTE: contours, IDs and rp will be updated
        # on gui_mouseReleaseEventImg2
        if left_click and self.eraserButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            Y, X = self.lab.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates()
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

        # Delete ID with middle click

        # Erase entire ID (set to 0)
        elif mid_click:
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
            self.update_IDsContours(prev_IDs, erasedIDs=[delID])

        # Separate bud
        elif right_click and self.separateBudButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            ID = self.lab[ydata, xdata]
            if ID == 0:
                # self.separateBudButton.setChecked(False)
                return
            # Store undo state before modifying stuff
            self.storeUndoRedoStates()
            max_ID = self.lab.max()

            # Get a mask of the object prior separation to later determine
            # the ID of the new seaprated object
            self.lab, success = self.auto_separate_bud_ID(
                                         ID, self.lab, self.rp,
                                         max_ID, enforce=True)

            # If automatic bud separation was not successfull call manual one
            if not success:
                paint_out = core.my_paint_app(
                                self.lab, ID, self.rp, del_small_obj=True,
                                overlay_img=self.img2.image)
                if paint_out.cancel:
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

            editID = apps.editID_widget(old_ID=ID)
            if editID.cancel:
                self.editID_Button.setChecked(False)
                return

            # Store undo state before modifying stuff
            self.storeUndoRedoStates()
            prev_IDs = [obj.label for obj in self.rp]
            for old_ID, new_ID in editID.new_ID:
                if new_ID in prev_IDs:
                    tempID = self.lab.max() + 1
                    self.lab[self.lab == old_ID] = tempID
                    self.lab[self.lab == new_ID] = old_ID
                    self.lab[self.lab == tempID] = new_ID

                    # Clear labels IDs of the swapped IDs
                    old_ID_idx = prev_IDs.index(old_ID)
                    new_ID_idx = prev_IDs.index(new_ID)
                    self.plot1.removeItem(self.plot1_items[old_ID_idx][0])
                    self.plot1.removeItem(self.plot1_items[new_ID_idx][0])
                    self.plot2.removeItem(self.plot2_items[old_ID_idx])
                    self.plot2.removeItem(self.plot2_items[new_ID_idx])
                    self.drawID_and_Contour(
                        old_ID_idx, self.rp[old_ID_idx],
                        drawContours=False
                    )
                    self.drawID_and_Contour(
                        new_ID_idx, self.rp[new_ID_idx],
                        drawContours=False
                    )

                    # Append information for replicating the edit in tracking
                    # List of tuples (y, x, replacing ID)
                    obj = self.rp[old_ID_idx]
                    y, x = obj.centroid
                    self.editID_info.append((y, x, new_ID))
                    obj = self.rp[new_ID_idx]
                    y, x = obj.centroid
                    self.editID_info.append((y, x, old_ID))
                else:
                    self.lab[self.lab == old_ID] = new_ID
                    # Clear labels IDs of the swapped IDs
                    old_ID_idx = prev_IDs.index(old_ID)
                    self.plot1.removeItem(self.plot1_items[old_ID_idx][0])
                    self.plot2.removeItem(self.plot2_items[old_ID_idx])
                    self.rp[old_ID_idx].label = new_ID
                    self.drawID_and_Contour(
                        old_ID_idx, self.rp[old_ID_idx],
                        drawContours=False
                    )
                    # Append information for replicating the edit in tracking
                    # List of tuples (y, x, replacing ID)
                    obj = self.rp[old_ID_idx]
                    y, x = obj.centroid
                    self.editID_info.append((y, x, new_ID))


            # Update rps
            self.update_rp()

            # Since we manually changed an ID we don't want to repeat tracking
            self.checkIDs_LostNew()

            # Update colors for the edited IDs
            self.updateLookuptable()

            self.img2.setImage(self.lab)
            self.editID_Button.setChecked(False)

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



    def gui_mouseDragEventImg2(self, event):
        # Eraser dragging mouse --> keep erasing
        if self.isMouseDragImg2:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(round(x)), int(round(y))
            brushSize = self.brushSizeSpinbox.value()
            mask = skimage.morphology.disk(brushSize, dtype=np.bool)
            ymin, xmin = ydata-brushSize, xdata-brushSize
            ymax, xmax = ydata+brushSize+1, xdata+brushSize+1
            self.erasedIDs.extend(self.lab[ymin:ymax, xmin:xmax][mask])
            self.lab[ymin:ymax, xmin:xmax][mask] = 0
            self.img2.updateImage()

    def gui_mouseReleaseEventImg2(self, event):
        # Eraser mouse release --> update IDs and contours
        if self.isMouseDragImg2:
            self.isMouseDragImg2 = False
            erasedIDs = np.unique(self.erasedIDs)

            # Update data (rp, etc)
            prev_IDs = [obj.label for obj in self.rp]
            self.update_rp()

            self.update_IDsContours(
                prev_IDs, erasedIDs=erasedIDs, newIDs=erasedIDs
            )

        # Merge IDs
        if self.mergeIDsButton.isChecked():
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
                prev_IDs, erasedIDs=[ID, self.firstID], newIDs=[newID]
            )
            self.mergeIDsButton.setChecked(False)


    def gui_mouseReleaseEventImg1(self, event):
        pass



    def gui_mousePressEventImg1(self, event):
        right_click = event.button() == Qt.MouseButton.RightButton
        left_click = event.button() == Qt.MouseButton.LeftButton
        # Paint new IDs with brush and left click on the right image
        if left_click and self.brushButton.isChecked():
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
                lab[ymin:ymax, xmin:xmax][mask] = self.brushID

                # Update data (rp, etc)
                prev_IDs = [obj.label for obj in self.rp]
                self.update_rp()

                # Repeat tracking
                self.tracking()

                newIDs = [self.lab[ydata, xdata]]

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


    def gui_mouseDragEventImg1(self, event):
        pass

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
                try:
                    self.wcLabel.setText(
                            f'(x={x:.2f}, y={y:.2f}, value={val:.2f})'
                    )
                except:
                    val = [v for v in val]
                    self.wcLabel.setText(
                            f'(x={x:.2f}, y={y:.2f}, value={val})'
                    )
            else:
                self.wcLabel.setText(f'')
        except:
            traceback.print_exc()
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
                    self.brushCircle.setData([x], [y],
                                                   size=size)
            else:
                self.brushCircle.setData([], [])
        except:
            # traceback.print_exc()
            self.brushCircle.setData([], [])

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
                self.wcLabel.setText(f'(x={x:.2f}, y={y:.2f}, value={val:.0f})')
            else:
                self.wcLabel.setText(f'')
        except:
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

        self.overlayButton = QToolButton(self)
        self.overlayButton.setIcon(QIcon(":overlay.svg"))
        self.overlayButton.setCheckable(True)
        self.overlayButton.setToolTip('Overlay fluorescent image')
        navigateToolBar.addWidget(self.overlayButton)

        # fluorescent image color widget
        self.colorButton = pg.ColorButton(self, color=(230,230,230))
        self.colorButton.setFixedHeight(32)
        self.colorButton.setDisabled(True)
        self.colorButton.setToolTip('Fluorescent image color')
        navigateToolBar.addWidget(self.colorButton)

        self.navigateToolBar = navigateToolBar



        # Edit toolbar
        editToolBar = QToolBar("Edit", self)
        # editToolBar.setFixedHeight(72)
        editToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(editToolBar)

        self.slideshowButton = QToolButton(self)
        self.slideshowButton.setIcon(QIcon(":eye-plus.svg"))
        self.slideshowButton.setCheckable(True)
        self.slideshowButton.setShortcut('Ctrl+W')
        self.slideshowButton.setToolTip('Open slideshow (Ctrl+W)')
        editToolBar.addWidget(self.slideshowButton)

        self.brushButton = QToolButton(self)
        self.brushButton.setIcon(QIcon(":brush.svg"))
        self.brushButton.setCheckable(True)
        self.brushButton.setShortcut('b')
        self.brushButton.setToolTip('Paint (b)')
        editToolBar.addWidget(self.brushButton)

        self.eraserButton = QToolButton(self)
        self.eraserButton.setIcon(QIcon(":eraser.png"))
        self.eraserButton.setCheckable(True)
        self.eraserButton.setShortcut('x')
        self.eraserButton.setToolTip('Erase (x)')
        editToolBar.addWidget(self.eraserButton)

        self.editID_Button = QToolButton(self)
        self.editID_Button.setIcon(QIcon(":edit-id.svg"))
        self.editID_Button.setCheckable(True)
        self.editID_Button.setShortcut('n')
        self.editID_Button.setToolTip('Edit ID (N + right-click)')
        editToolBar.addWidget(self.editID_Button)

        self.separateBudButton = QToolButton(self)
        self.separateBudButton.setIcon(QIcon(":separate-bud.svg"))
        self.separateBudButton.setCheckable(True)
        self.separateBudButton.setShortcut('s')
        self.separateBudButton.setToolTip('Separate bud (S + right-click)')
        editToolBar.addWidget(self.separateBudButton)

        self.mergeIDsButton = QToolButton(self)
        self.mergeIDsButton.setIcon(QIcon(":merge-IDs.svg"))
        self.mergeIDsButton.setCheckable(True)
        self.mergeIDsButton.setShortcut('m')
        self.mergeIDsButton.setToolTip('Merge IDs (S + right-click)')
        editToolBar.addWidget(self.mergeIDsButton)

        self.binCellButton = QToolButton(self)
        self.binCellButton.setIcon(QIcon(":bin.svg"))
        self.binCellButton.setCheckable(True)
        self.binCellButton.setToolTip(
           "Annotate cell as 'Removed from analysis' (R + right-click)"
        )
        self.binCellButton.setShortcut("r")
        editToolBar.addWidget(self.binCellButton)

        self.ripCellButton = QToolButton(self)
        self.ripCellButton.setIcon(QIcon(":rip.svg"))
        self.ripCellButton.setCheckable(True)
        self.ripCellButton.setToolTip(
           "Annotate cell as dead (D + right-click)"
        )
        self.ripCellButton.setShortcut("d")
        editToolBar.addWidget(self.ripCellButton)

        editToolBar.addAction(self.repeatTrackingAction)

        self.disableTrackingCheckBox = QCheckBox("Disable tracking")
        self.disableTrackingCheckBox.setLayoutDirection(Qt.RightToLeft)
        editToolBar.addWidget(self.disableTrackingCheckBox)

        self.brushSizeSpinbox = QSpinBox()
        self.brushSizeSpinbox.setValue(3)
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

    def setEnabledToolbarButton(self, enabled=False):
        self.showInExplorerAction.setEnabled(enabled)
        self.reloadAction.setEnabled(enabled)
        self.saveAction.setEnabled(enabled)
        self.editToolBar.setEnabled(enabled)
        self.navigateToolBar.setEnabled(enabled)
        self.modeToolBar.setEnabled(enabled)
        self.enableSizeSpinbox(False)

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
        self.app.restoreOverrideCursor()

    def changeMode(self, mode):
        if mode == 'Segmentation and Tracking':
            self.setEnabledToolbarButton(enabled=True)
            self.disableTrackingCheckBox.setChecked(False)
        else:
            self.setEnabledToolbarButton(enabled=False)
            self.navigateToolBar.setEnabled(True)
            self.modeToolBar.setEnabled(True)
            self.disableTrackingCheckBox.setChecked(True)

    def launchSlideshow(self):
        if self.slideshowButton.isChecked():
            self.slideshowWin = apps.CellsSlideshow_GUI(
                                   button_toUncheck=self.slideshowButton)
            self.slideshowWin.loadData(self.data.img_data, frame_i=self.frame_i)
            self.slideshowWin.show()
        else:
            self.slideshowWin = None

    def nearest_nonzero(self, a, y, x):
        r, c = np.nonzero(a)
        dist = ((r - y)**2 + (c - x)**2)
        min_idx = dist.argmin()
        return a[r[min_idx], c[min_idx]]

    def convexity_defects(self, img, eps_percent):
        img = img.astype(np.uint8)
        contours, hierarchy = cv2.findContours(img,2,1)
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
                    # curr_ID_bud = IDs_sep[areas.index(min(areas))]
                    # curr_ID_moth = IDs_sep[areas.index(max(areas))]
                    orig_sblab = np.copy(sep_bud_label)
                    # sep_bud_label = np.zeros_like(sep_bud_label)
                    # sep_bud_label[orig_sblab==curr_ID_moth] = ID
                    # sep_bud_label[orig_sblab==curr_ID_bud] = max(IDs)+max_i
                    sep_bud_label *= (max_ID+max_i)
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
        self.brushCircle.setSize(self.brushSizeSpinbox.value()*2)


    def Brush_cb(self, event):
        if self.eraserButton.isChecked():
            self.eraserButton.toggled.disconnect()
            self.eraserButton.setChecked(False)
            self.eraserButton.toggled.connect(self.Eraser_cb)
        if not self.brushButton.isChecked():
            self.brushCircle.setData([], [])
            self.enableSizeSpinbox(False)
            # self.app.restoreOverrideCursor()
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
        img = skimage.exposure.equalize_adapthist(self.img1.image)
        self.img1.setImage(img)


    def Eraser_cb(self, event):
        if self.brushButton.isChecked():
            self.brushButton.toggled.disconnect()
            self.brushButton.setChecked(False)
            self.brushButton.toggled.connect(self.Brush_cb)
        if not self.eraserButton.isChecked():
            self.brushCircle.setData([], [])
            self.brushButton.setChecked(False)
            self.enableSizeSpinbox(False)
            # self.app.restoreOverrideCursor()
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
        # elif ev.key() == Qt.Key_Plus:
        #     self.storeUndoRedoStates()
        # elif ev.key() == Qt.Key_Right:
        #     self.next_cb()
        # elif ev.key() == Qt.Key_Left:
        #     self.prev_cb()
        # elif ev.text() == 'b':
        #     self.BrushEraser_cb(ev)

    def storeUndoRedoStates(self):
        # Since we modified current frame all future frames that were already
        # visited are not valid anymore. Undo changes there
        self.undo_changes_future_frames()

        # Restart count from the most recent state (index 0)
        # NOTE: index 0 is most recent state before doing last change
        self.UndoCount = 0
        self.undoAction.setEnabled(True)
        self.UndoRedoStates.insert(0, {'labels': self.lab.copy()})
        # Keep only 5 Undo/Redo states
        if len(self.UndoRedoStates) > 5:
            self.UndoRedoStates.pop(-1)

    def undo(self):
        self.lab = self.UndoRedoStates[self.UndoCount]['labels']
        self.update_rp()
        self.updateALLimg()
        if self.UndoCount < len(self.UndoRedoStates)-1:
            self.UndoCount += 1
            # Since we have undone then it is possible to redo
            self.redoAction.setEnabled(True)
        else:
            # We have undone all available states
            self.undoAction.setEnabled(False)


    def redo(self):
        self.lab = self.UndoRedoStates[self.UndoCount]['labels']
        self.rp = skimage.measure.regionprops(self.lab)
        self.updateALLimg()
        if self.UndoCount > 0:
            self.UndoCount -= 1
            # Since we have redone then it is possible to undo
            self.undoAction.setEnabled(True)
        else:
            # We have redone all available states
            self.redoAction.setEnabled(False)



    def disableTracking(self):
        pass

    def repeatTracking(self):
        self.tracking(enforce=True)
        self.updateALLimg()

    def repeatSegmYeaZ(self):
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
        lab = segment.segment(thresh, pred, min_distance=5).astype(int)
        self.is_first_call_YeaZ = False
        self.lab = lab
        self.update_rp()
        self.tracking()
        self.updateALLimg()

    def repeatSegmCellpose(self):
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

    def next_cb(self):
        self.app.setOverrideCursor(Qt.WaitCursor)
        if self.frame_i < self.num_frames-1:
            # Store data for current frame
            self.store_data(debug=True)
            # Go to next frame
            self.frame_i += 1
            self.get_data()
            self.tracking()
            self.updateALLimg()
        else:
            print('You reached the last frame!')
        if self.slideshowWin is not None:
            self.slideshowWin.frame_i = self.frame_i
            self.slideshowWin.update_img()
        self.app.restoreOverrideCursor()

    def prev_cb(self):
        self.app.setOverrideCursor(Qt.WaitCursor)
        if self.frame_i > 0:
            self.frame_i -= 1
            self.get_data()
            self.tracking()
            self.updateALLimg()
        else:
            print('You reached the first frame!')
        if self.slideshowWin is not None:
            self.slideshowWin.frame_i = self.frame_i
            self.slideshowWin.update_img()
        self.app.restoreOverrideCursor()

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
        if self.data.last_tracked_i is not None:
            self.app.restoreOverrideCursor()
            last_tracked_num = self.data.last_tracked_i+1
            msg = QtGui.QMessageBox()
            start_from_last_tracked_i = msg.question(
                self, 'Start from last session?',
                'The system detected a previous session ended '
                f'at frame {last_tracked_num}.\n\n'
                f'Do you want to resume from frame {last_tracked_num}?',
                msg.Yes | msg.No
            )
            if start_from_last_tracked_i == msg.Yes:
                for i in range(last_tracked_num):
                    self.frame_i = i
                    self.get_data()
                    self.update_rp_metadata(draw=False)
                    self.store_data()
                    self.binnedIDs = set()
                    self.ripIDs = set()
            else:
                self.frame_i = 0

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

    def store_data(self, debug=False):
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
                        'y_centroid': yy_centroid,
            }
        ).set_index('Cell_ID')

    def get_data(self):
        self.UndoRedoStates = []
        self.UndoCount = 0
        self.editID_info = []
        # If stored labes is None then it is the first time we visit this frame
        if self.allData_li[self.frame_i]['labels'] is None:
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
        else:
            # Requested frame was already visited. Load from RAM.
            self.lab = self.allData_li[self.frame_i]['labels'].copy()
            self.rp = skimage.measure.regionprops(self.lab)
            df = self.allData_li[self.frame_i]['segm_metadata_df']
            binnedIDs_df = df[df['is_cell_excluded']]
            self.binnedIDs = set(binnedIDs_df.index)
            ripIDs_df = df[df['is_cell_dead']]
            self.ripIDs = set(ripIDs_df.index)

    def drawID_and_Contour(self, i, obj, drawContours=True):
        y, x = obj.centroid
        _IDlabel1 = pg.LabelItem(
                text=f'{obj.label}',
                color='FA0000',
                bold=True,
                size='10pt'
        )
        w, h = _IDlabel1.rect().right(), _IDlabel1.rect().bottom()
        _IDlabel1.setPos(x-w/2, y-h/2)
        self.plot1.addItem(_IDlabel1)

        _IDlabel2 = pg.LabelItem(
                text=f'{obj.label}',
                color='FA0000',
                bold=True,
                size='10pt'
        )
        w, h = _IDlabel2.rect().right(), _IDlabel2.rect().bottom()
        _IDlabel2.setPos(x-w/2, y-h/2)
        self.plot2.addItem(_IDlabel2)
        self.plot2_items[i] = _IDlabel2

        if not drawContours:
            self.plot1_items[i] = (_IDlabel1, self.plot1_items[i][1])
            return

        contours, hierarchy = cv2.findContours(
                                           obj.image.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
        min_y, min_x, _, _ = obj.bbox
        try:
            cont = np.squeeze(contours[0], axis=1)
        except:
            qt_debug_trace()
        cont = np.vstack((cont, cont[0]))
        cont += [min_x, min_y]
        cont_plot = self.plot1.plot(cont[:,0], cont[:,1], pen=self.cpen)

        self.plot1_items[i] = (_IDlabel1, cont_plot)

    def update_rp(self):
        # Update rp for current self.lab (e.g. after any change)
        self.rp = skimage.measure.regionprops(self.lab)
        self.update_rp_metadata()

    def update_IDsContours(self, prev_IDs, erasedIDs=[],
                           newIDs=[]):
        """Function to draw labels text and contours of specific IDs.
        It should speed up things because we draw only few IDs usually.

        Parameters
        ----------
        prev_IDs : list
            List of IDs before the change (e.g. before erasing ID or painting
            a new one etc.)
        erasedIDs : list
            List of IDs that needs to be RE-drawn. An erased ID will be first
            removed and then RE-drawn with the new label and contour
        newIDs : bool
            List of new IDs that are present after a change.

        Returns
        -------
        None.

        """

        # Remove contours and ID label for erased IDs
        for erasedID in erasedIDs:
            if erasedID in prev_IDs:
                erased_idx = prev_IDs.index(erasedID)
                if self.plot1_items[erased_idx] is not None:
                    self.plot1.removeItem(self.plot1_items[erased_idx][0])
                    self.plot1.removeItem(self.plot1_items[erased_idx][1])
                if self.plot2_items[erased_idx] is not None:
                    self.plot2.removeItem(self.plot2_items[erased_idx])

        if len(newIDs)>0:
            for i, obj in enumerate(self.rp):
                ID = obj.label
                if ID in newIDs:
                    self.plot1_items.insert(i, None)
                    self.plot2_items.insert(i, None)
                    # Draw ID labels and contours of new objects
                    self.drawID_and_Contour(i, obj)

    def updateLookuptable(self):
        lut = self.lut[:self.lab.max()+1].copy()
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
                    LabelID = self.plot2_items[i]
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
                    LabelID = self.plot2_items[i]
                    LabelID.setText(f'{ID}', color=(150, 0, 0))
                    ripIDs_xx.append(x)
                    ripIDs_yy.append(y)
            else:
                obj.dead = False

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
                self.app.restoreOverrideCursor()
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
                    self.app.restoreOverrideCursor()
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
            self.app.restoreOverrideCursor()

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

    def updateALLimg(self):
        self.frameLabel.setText(
                 f'Current frame = {self.frame_i+1}/{self.num_frames}')

        if self.overlayButton.isChecked():
            cells_img = self.data.img_data[self.frame_i]
            fluo_img = self.data.ol_frames[self.frame_i]
            img = self.get_overlay(fluo_img, cells_img)
        else:
            img = self.data.img_data[self.frame_i]

        lab = self.lab

        self.img1.setImage(img)
        self.img2.setImage(lab)
        self.updateLookuptable()

        # Remove previous IDs texts and contours
        for _item in self.plot1_items:
            if _item is not None:
                IDlabel, cont = _item
                self.plot1.removeItem(IDlabel)
                self.plot1.removeItem(cont)

        for _item in self.plot2_items:
            self.plot2.removeItem(_item)

        self.plot1_items = [None]*len(self.rp)
        self.plot2_items = [None]*len(self.rp)
        # Annotate cell ID and draw contours
        for i, obj in enumerate(self.rp):
            self.drawID_and_Contour(i, obj)

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


    def tracking(self, onlyIDs=[], enforce=False):
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

        allIDs = tracked_IDs.extend(new_tracked_IDs_2)
        # Correct tracking with manually changed IDs
        for y, x, new_ID in self.editID_info:
            old_ID = self.lab[y, x]
            if new_ID in allIDs:
                tempID = self.lab.max() + 1
                self.lab[self.lab == old_ID] = tempID
                self.lab[self.lab == new_ID] = old_ID
                self.lab[self.lab == tempID] = new_ID
            else:
                self.lab[self.lab == old_ID] = new_ID

        # Update labels, regionprops and determine new and lost IDs
        self.lab = tracked_lab
        self.update_rp()
        self.checkIDs_LostNew()

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

        if not values:
            txt = (
                'The selected folder:\n\n '
                f'{exp_path}\n\n'
                'is not a valid folder. Select a folder that contains the Position_n folders'
            )
            msg = QtGui.QMessageBox()
            msg.critical(
                self, 'Incompatible folder', txt, msg.Ok
            )
            return

        pos_foldername = select_folder.run_widget(values, allow_abort=False)

        if select_folder.was_aborted:
            self.titleLabel.setText('File --> Open to start the process')
            return

        images_path = f'{exp_path}/{pos_foldername}/Images'

        self.images_path = images_path

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

        self.app.setOverrideCursor(Qt.WaitCursor)
        print(f'Loading {img_path}...')

        self.init_frames_data(img_path, user_ch_name)

        # Connect events at the end of loading data process
        self.gui_connectGraphicsEvents()
        self.gui_connectEditActions()

        self.titleLabel.setText(
                'Data successfully loaded. Right/Left arrow to navigate frames')

        self.app.restoreOverrideCursor()

    def showInExplorer(self):
        systems = {
            'nt': os.startfile,
            'posix': lambda foldername: os.system('xdg-open "%s"' % foldername),
            'os2': lambda foldername: os.system('open "%s"' % foldername)
             }

        systems.get(os.name, os.startfile)(self.images_path)

    def add_static_metadata_df(self, df, rp):
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
            segm_metadata_csv_path = self.data.segm_metadata_csv_path
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

            all_frames_metadata_df = pd.concat(
                df_li, keys=keys, names=['frame_i', 'Cell_ID']
            )

            # Save segmentation metadata
            all_frames_metadata_df.to_csv(segm_metadata_csv_path)

            # Save segmentation file
            np.save(segm_npy_path, segm_npy)
            # Save last tracked frame
            with open(last_tracked_i_path, 'w+') as txt:
                txt.write(str(frame_i))

            print('--------------')
            print(f'Saved data until frame number {frame_i}')
            print('--------------')

            self.app.restoreOverrideCursor()

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

    def closeEvent(self, event):
        if self.slideshowWin is not None:
            self.slideshowWin.close()
        if self.editToolBar.isEnabled():
            msg = QtGui.QMessageBox()
            save = msg.question(
                self, 'Save?', 'Do you want to save segmentation data?',
                msg.Yes | msg.No
            )
            if save == msg.Yes:
                self.saveFile()

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
