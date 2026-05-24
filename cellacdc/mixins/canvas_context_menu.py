"""View adapter for canvas context menus and deleted-ROI clicks."""

from __future__ import annotations

import pyqtgraph as pg
from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QAction, QMenu

from .image_display import ImageDisplay

class CanvasContextMenu(ImageDisplay):
    """Extracted from guiWin."""

    def gui_clickedDelRoi(self, event, left_click, right_click):
        posData = self.data[self.pos_i]
        x, y = event.pos().x(), event.pos().y()

        # Check if right click on ROI
        delROIs = (
            posData.allData_li[posData.frame_i]['delROIs_info']['rois'].copy()
        )
        for r, roi in enumerate(delROIs):
            ROImask = self.getDelRoiMask(roi)
            if self.isSegm3D:
                clickedOnROI = ROImask[self.z_lab(), int(y), int(x)]
            else:
                clickedOnROI = ROImask[int(y), int(x)]
            raiseContextMenuRoi = right_click and clickedOnROI
            dragRoi = left_click and clickedOnROI
            if raiseContextMenuRoi:
                self.roi_to_del = roi
                self.roiContextMenu = QMenu(self)
                separator = QAction(self)
                separator.setSeparator(True)
                self.roiContextMenu.addAction(separator)
                action = QAction('Remove ROI')
                action.triggered.connect(self.removeDelROI)
                self.roiContextMenu.addAction(action)
                try:
                    # Convert QPointF to QPoint
                    self.roiContextMenu.exec_(event.screenPos().toPoint())
                except AttributeError:
                    self.roiContextMenu.exec_(event.screenPos())
                return True
            elif dragRoi:
                event.ignore()
                return True
        return False

    def checkHighlightScaleBar(self, x, y, activeToolButton):
        if not hasattr(self, 'scaleBar'):
            return
        
        if not self.addScaleBarAction.isChecked():
            return
        
        if activeToolButton is not None:
            return
        
        ymin, xmin, ymax, xmax = self.scaleBar.bbox()
        if x < xmin:
            self.scaleBar.setHighlighted(False)
            return
        
        if x > xmax:
            self.scaleBar.setHighlighted(False)
            return
        
        if y < ymin:
            self.scaleBar.setHighlighted(False)
            return
        
        if y > ymax:
            self.scaleBar.setHighlighted(False)
            return

        self.scaleBar.setHighlighted(True)

    def checkHighlightTimestamp(self, x, y, activeToolButton):
        if not hasattr(self, 'timestamp'):
            return
        
        if not self.addTimestampAction.isChecked():
            return
        
        if activeToolButton is not None:
            return
        
        if hasattr(self, 'scaleBar'):
            if self.scaleBar.isHighlighted():
                return
        
        ymin, xmin, ymax, xmax = self.timestamp.bbox()
        if x < xmin:
            self.timestamp.setHighlighted(False)
            return
        
        if x > xmax:
            self.timestamp.setHighlighted(False)
            return
        
        if y < ymin:
            self.timestamp.setHighlighted(False)
            return
        
        if y > ymax:
            self.timestamp.setHighlighted(False)
            return

        self.timestamp.setHighlighted(True)

    def gui_imgGradShowContextMenu(self, x, y):
        if hasattr(self, 'scaleBar'):
            if self.scaleBar.isHighlighted():
                self.scaleBar.showContextMenu(x, y)
                return
        
        if hasattr(self, 'timestamp'):
            if self.timestamp.isHighlighted():
                self.timestamp.showContextMenu(x, y)
                return
            
        self.imgGrad.gradient.menu.popup(QPoint(int(x), int(y)))

    def gui_rightImageShowContextMenu(self, event):
        try:
            # Convert QPointF to QPoint
            self.imgGradRight.gradient.menu.popup(event.screenPos().toPoint())
        except AttributeError:
            self.imgGradRight.gradient.menu.popup(event.screenPos())
