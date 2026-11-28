"""Deletion ROIs and the label ROI / freehand region tool of the main GUI.
Mixins of guiWin (gui/main_window/window.py) only: the measurements
viewer never composes them.
"""
import os
import uuid
from functools import partial
import numpy as np
import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.draw
import skimage.exposure
import skimage.transform
import skimage.segmentation
from qtpy.QtCore import (
    Qt, QRect, QRectF, QTimer, QThread, QMutex, QWaitCondition
)
from qtpy.QtGui import QCursor, QGuiApplication
from qtpy.QtWidgets import QAction, QMenu
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import exception_handler
from cellacdc import html_utils
from cellacdc.shared.dialogs import model_selection as dialogs_model_selection
from ..dialogs import segmentation as gui_dialogs_segmentation
from ..workers import segmentation_tracking as gui_workers_segmentation_tracking
from cellacdc.shared.widgets import message_box as widgets_message_box
from ..widgets import main_window_items as gui_widgets_main_window_items
from cellacdc import settings_folderpath
from cellacdc import qutils
from cellacdc import transformation


class DeleteRoisGuiElements:
    """Deletion ROIs (rectangle and poly-line): creation, preview,
    propagation to future frames and application.
    """

    def isMouseOnDelRoi(self, roi, x, y):
        for handle in roi.handles:
            h = handle['item']
            if h.currentPen is h.hoverPen:
                return True

        mask, bbox = self.getRoiCoords(roi, return_mask=True)
        # check if click insde bbox by correcting x and y and checking if something is out of bounds
        (r0, r1, c0, c1) = bbox
        if y<r0 or y>=r1 or x<c0 or x>=c1:
            return False

        # exact check
        y_rel, x_rel = int(y-r0), int(x-c0)
        if mask[y_rel, x_rel]:
            return True
        
        return False

    def gui_handleClickOnDelRoi(self, event, left_click, right_click):
        posData = self.data[self.pos_i]
        x, y = event.pos().x(), event.pos().y()

        # Check if click on ROI
        delROIs = (
            posData.allData_li[posData.frame_i]['delROIs_info']['rois']
        )
        for r, roi in enumerate(delROIs):
            clickedOnROI = self.isMouseOnDelRoi(roi, x, y)
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

    def gui_getHoveredSegmentsPolyLineRoi(self):
        posData = self.data[self.pos_i]
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        segments = []
        for roi in delROIs_info['rois']:
            if not isinstance(roi, pg.PolyLineROI):
                continue       
            for seg in roi.segments:       
                if seg.currentPen == seg.hoverPen:
                    seg.roi = roi
                    segments.append(seg)
        return segments

    def gui_getHoveredHandlesPolyLineRoi(self):
        posData = self.data[self.pos_i]
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        handles = []
        for roi in delROIs_info['rois']:
            if not isinstance(roi, pg.PolyLineROI):
                continue           
            for handle in roi.getHandles():       
                if handle.currentPen == handle.hoverPen:
                    handle.roi = roi
                    handles.append(handle)
        return handles

    def gui_isHoveringRoiHandle(self, roi):
        if roi is None:
            return False
        try:
            handles = roi.getHandles()
        except Exception:
            return False

        for handle in handles:
            try:
                if handle.currentPen == handle.hoverPen:
                    return True
            except Exception:
                continue
        return False

    def gui_isActiveInterceptRoi(self, roi):
        if roi is None:
            return False
        if roi is getattr(self, 'mergeIDsRoiItem', None):
            return getattr(self, '_isMergeIDsRoiVisible', False)
        if roi is getattr(self, 'whitelistIDsRoiItem', None):
            return getattr(self, '_isWhitelistIDsRoiVisible', False)
        return True

    def gui_isMouseInsideRoi(self, roi, event):
        if roi is None:
            return False
        try:
            ymin, xmin, ymax, xmax = roi.bbox()
        except Exception:
            return False

        if xmin == xmax or ymin == ymax:
            return False

        x, y = event.pos().x(), event.pos().y()
        return xmin <= x <= xmax and ymin <= y <= ymax

    def removeAlldelROIsCurrentFrame(self):
        posData = self.data[self.pos_i]
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        rois = delROIs_info['rois'].copy()
        for roi in rois:
            self.ax2.removeDelRoiItem(roi)
            self.ax1.removeDelRoiItem(roi)

        # Only deletion ROIs (they have a `key`): e.g., the box prompts of
        # promptable models (`widgets.PromptBoxROI`) are other ROIs. Iterate
        # over copies because the items are removed from the lists
        for item in list(self.ax2.items):
            if isinstance(item, pg.ROI) and hasattr(item, 'key'):
                self.ax2.removeDelRoiItem(item)

        for item in list(self.ax1.items):
            if item is self.labelRoiItem:
                continue
            if isinstance(item, pg.ROI) and hasattr(item, 'key'):
                self.ax1.removeDelRoiItem(item)
        # if self.polyLineRoi is shown also clear it
        
        if hasattr(self, 'polyLineRoi') and self.polyLineRoi is not None:
            try:
                self.polyLineRoi = None
                # self.polyLineRoi.clearPoints()
            except:
                pass

    def removeDelROI(self, event):
        posData = self.data[self.pos_i]
        
        for ax in (self.ax1, self.ax2):
            try:
                ax.removeDelRoiItem(self.roi_to_del)
            except Exception as err:
                pass
        
        self.applyDelROI(self.roi_to_del, removing_roi=True)
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        idx = delROIs_info['rois'].index(self.roi_to_del)
        
        delROIs_info['rois'].pop(idx)
        delROIs_info['delMasks'].pop(idx)
        delROIs_info['delIDsROI'].pop(idx)
        delROIs_info['state'].pop(idx)
        delROIs_info['delMasksCoords'].pop(idx)
        
        QTimer.singleShot(
            300, partial(self.removeDelROIFromFutureFrames, self.roi_to_del)
        )

    def removeDelROIFromFutureFrames(self, roi_to_del):
        posData = self.data[self.pos_i]
        
        # Restore deleted IDs from already visited future frames
        current_frame_i = posData.frame_i
        has_to_restore_frame_i = False
        for i in range(posData.frame_i+1, posData.SizeT):            
            delROIs_info = posData.allData_li[i]['delROIs_info']
            try:
                idx = delROIs_info['rois'].index(roi_to_del) 
            except ValueError:
                continue
            has_to_restore_frame_i = True
            self.store_data(autosave=False)
            posData.frame_i = i
            self.get_data()
            self.applyDelROI(roi_to_del, update_img=False, removing_roi=True)
            idx = delROIs_info['rois'].index(roi_to_del)
            delROIs_info['rois'].pop(idx)
            delROIs_info['delMasks'].pop(idx)
            delROIs_info['delIDsROI'].pop(idx)
            delROIs_info['state'].pop(idx)
            delROIs_info['delMasksCoords'].pop(idx)
        
        if isinstance(roi_to_del, pg.PolyLineROI):
            # PolyLine ROIs are only on ax1
            self.ax1.removeItem(roi_to_del)
        elif not self.labelsGrad.showLabelsImgAction.isChecked():
            # Rect ROI is on ax1 because ax2 is hidden
            self.ax1.removeItem(roi_to_del)
        else:
            # Rect ROI is on ax2 because ax2 is visible
            self.ax2.removeItem(roi_to_del)

        if not has_to_restore_frame_i:
            return
        
        # Back to current frame
        self.store_data()
        posData.frame_i = current_frame_i
        self.get_data()
        self.store_data()

    def updateDelROIinFutureFrames(self, roi: pg.ROI):
        posData = self.data[self.pos_i]
        restore_current_frame = False
        current_frame_i = posData.frame_i

        roiState = roi.getState()
        self.store_data()
        
        for i in range(posData.frame_i+1, posData.SizeT):
            delROIs_info = posData.allData_li[i]['delROIs_info']
            try:
                idx = delROIs_info['rois'].index(roi)
            except ValueError:
                continue
            
            delROIs_info['state'][idx] = roiState
            if posData.allData_li[i]['labels'] is None:
                continue
            
            posData.frame_i = i
            self.get_data()
            # here we manually set frame_i
            self.applyDelROI(roi, update_img=False)
            self.store_data(autosave=False)
            restore_current_frame = True
        
        if not restore_current_frame:
            return
        
        # Back to current frame
        posData.frame_i = current_frame_i
        posData.lab = posData.allData_li[posData.frame_i]['labels']                   
        self.get_data()
        self.store_data()

    def addDelROI(self, event):       
        roi, key = self.createDelROI()
        self.addRoiToDelRoiInfo(roi)
        if not self.labelsGrad.showLabelsImgAction.isChecked():
            self.ax1.addDelRoiItem(roi, key)
        else:
            self.ax2.addDelRoiItem(roi, key)
            
        self.applyDelROI(roi, update_img=False)
        
        QTimer.singleShot(
            300, partial(self.updateDelROIinFutureFrames, roi)
        )

        if self.isSnapshot:
            self.fixCcaDfAfterEdit('Delete IDs using ROI')
            self.updateAllImages()
        else:
            self.warnEditingWithCca_df(
                'Delete IDs using ROI', get_cancelled=True
            )

    def replacePolyLineRoiWithLineRoi(self, roi):
        x0, y0 = roi.pos().x(), roi.pos().y()
        (_, point1), (_, point2) = roi.getLocalHandlePositions()
        xr1, yr1 = point1.x(), point1.y()
        xr2, yr2 = point2.x(), point2.y()
        x1, y1 = xr1+x0, yr1+y0
        x2, y2 = xr2+x0, yr2+y0
        key = self.polyLineRoi.key
        lineRoi = pg.LineROI((x1, y1), (x2, y2), width=0.5)
        lineRoi.handleSize = 7
        lineRoi.key = key
        self.ax1.removeItem(self.polyLineRoi)
        self.ax1.addItem(lineRoi)
        lineRoi.removeHandle(2)
        # Connect closed ROI
        lineRoi.sigRegionChangeStarted.connect(self.delROIstartedMoving)
        lineRoi.sigRegionChanged.connect(self.delROImoving)
        lineRoi.sigRegionChangeFinished.connect(self.delROImovingFinished)
        return lineRoi

    def addRoiToDelRoiInfo(self, roi: pg.ROI):
        posData = self.data[self.pos_i]
        for i in range(posData.frame_i, posData.SizeT):
            delROIs_info = posData.allData_li[i]['delROIs_info']
            delROIs_info['rois'].append(roi)
            delROIs_info['state'].append(roi.getState())
            delROIs_info['delMasks'].append(list())
            delROIs_info['delMasksCoords'].append(list())
            delROIs_info['delIDsROI'].append(list())

    def addDelPolyLineRoi_cb(self, checked):
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.addDelPolyLineRoiButton)
            self.connectLeftClickButtons()
            if self.isSnapshot:
                self.fixCcaDfAfterEdit('Delete IDs using ROI')
                self.updateAllImages()
            else:
                self.warnEditingWithCca_df('Delete IDs using ROI')
        else:
            self.tempSegmentON = False
            self.ax1_rulerPlotItem.setData([], [])
            self.ax1_rulerAnchorsItem.setData([], [])
            self.startPointPolyLineItem.setData([], [])
            while self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()

    def createDelPolyLineRoi(self, key=None):
        Y, X = self.currentLab2D.shape
        self.polyLineRoi = pg.PolyLineROI(
            [], rotatable=False,
            removable=True,
            pen=pg.mkPen(color='r')
        )
        self.polyLineRoi.handleSize = 7
        self.polyLineRoi.points = []
        self.polyLineRoi.setCursor(Qt.SizeAllCursor)
        if key is None:
            key = uuid.uuid4()
        self.ax1.addDelRoiItem(self.polyLineRoi, key)

    def addPointsPolyLineRoi(self, closed=False):
        self.polyLineRoi.setPoints(self.polyLineRoi.points, closed=closed)
        if not closed:
            return

        # Connect closed ROI
        self.polyLineRoi.sigRegionChangeStarted.connect(self.delROIstartedMoving)
        self.polyLineRoi.sigRegionChanged.connect(self.delROImoving)
        self.polyLineRoi.sigRegionChangeFinished.connect(self.delROImovingFinished)

    def getViewRange(self):
        Y, X = self.img1.image.shape[:2]
        xRange, yRange = self.ax1.viewRange()
        xmin = 0 if xRange[0] < 0 else xRange[0]
        ymin = 0 if yRange[0] < 0 else yRange[0]
        
        xmax = X if xRange[1] >= X else xRange[1]
        ymax = Y if yRange[1] >= Y else yRange[1]
        return int(ymin), int(ymax), int(xmin), int(xmax)

    def createDelROI(self, xl=None, yb=None, w=32, h=32, anchors=None, key=None):
        posData = self.data[self.pos_i]
        if xl is None:
            xRange, yRange = self.ax1.viewRange()
            xl = 0 if xRange[0] < 0 else xRange[0]
            yb = 0 if yRange[0] < 0 else yRange[0]
        Y, X = self.currentLab2D.shape
        if anchors is None:
            roi = gui_widgets_main_window_items.DelROI(
                [xl, yb], [w, h],
                rotatable=False,
                removable=True,
                pen=pg.mkPen(color='r'),
                maxBounds=QRectF(QRect(0,0,X,Y))
            )
            ## handles scaling horizontally around center
            roi.addScaleHandle([1, 0.5], [0, 0.5])
            roi.addScaleHandle([0, 0.5], [1, 0.5])

            ## handles scaling vertically from opposite edge
            roi.addScaleHandle([0.5, 0], [0.5, 1])
            roi.addScaleHandle([0.5, 1], [0.5, 0])

            ## handles scaling both vertically and horizontally
            roi.addScaleHandle([1, 1], [0, 0])
            roi.addScaleHandle([0, 0], [1, 1])
            roi.addScaleHandle([0, 1], [1, 0])
            roi.addScaleHandle([1, 0], [0, 1])

        roi.handleSize = 7
        roi.sigRegionChangeStarted.connect(self.delROIstartedMoving)
        roi.sigRegionChanged.connect(self.delROImoving)
        roi.sigRegionChangeFinished.connect(self.delROImovingFinished)
        
        if key is None:
            key = uuid.uuid4()
        
        return roi, key

    def delROIstartedMoving(self, roi):
        if self.draggingDelROI is True: # already ran started moving routine
            return

        self.draggingDelROI = True
        self.delROIpreviewDelIDsObjDict = dict()
        self.delROIoriginalLab = self.currentLab2D.copy()
        # placeholder readd weakly deleted items

    def delROImoving(self, roi):
        self.updateROIpreview(roi)


    def _getROIDelIDs(self, roi, lab, hidden_IDs):        
        mask, bbox = self.getRoiCoords(roi, return_mask=True)
        cutout = np.where(mask, lab[bbox[0]:bbox[1], bbox[2]:bbox[3]], 0)   # same shape as mask, non-roi pixels set to 0
            
        touching_IDs = set(self._acdc_regionprops.find_IDs(cutout))
        to_hide_IDs = touching_IDs - set(hidden_IDs)
        to_restore_IDs = set(hidden_IDs) - touching_IDs
        
        return to_hide_IDs, to_restore_IDs

    def updateROIpreview(self, roi):
        if not self.draggingDelROI:
            return
        
        hidden_IDs = self.delROIpreviewDelIDsObjDict.keys()
        lab = self.delROIoriginalLab
        to_hide_IDs, to_restore_IDs = self._getROIDelIDs(roi, lab, hidden_IDs)
        
        if not to_hide_IDs and not to_restore_IDs:
            return
        
        posData = self.data[self.pos_i]
        curr_lab_view = self.currentLab2D
        for ID in to_hide_IDs:
            rp = self.rpCurr2D()
            obj = rp.get_obj_from_ID(ID)
            self.delROIpreviewDelIDsObjDict[ID] = obj
            curr_lab_view[obj.slice][obj.image] = 0
        
        if self.isContoursChecked(0):
            self.updateContoursImage(ax=0, delROIsIDs=hidden_IDs)
        if self.isContoursChecked(1):
            self.updateContoursImage(ax=1, delROIsIDs=hidden_IDs)
                    
        for ID in to_restore_IDs:
            obj = self.delROIpreviewDelIDsObjDict.pop(ID)
            curr_lab_view[obj.slice][obj.image] = ID

            if self.isContoursChecked(0):
                self.addObjContourToContoursImage(
                    obj=obj, ax=0
                )
            if self.isContoursChecked(1):
                self.addObjContourToContoursImage(
                    obj=obj, ax=1
                )

        if self.isOverlaySegmMaskChecked(0):
            self.labelsLayerImg1.setImage(
               curr_lab_view, autoLevels=False
            )
        if self.isOverlaySegmMaskChecked(1):
            self.labelsLayerRightImg.setImage(
                curr_lab_view, autoLevels=False
            )

    def getRoiPixelCoords(self, roi):
        """Raw (rr, cc) pixel coords covered by roi, no z/mask handling."""
        if isinstance(roi, pg.PolyLineROI):
            r, c = [], []
            x0, y0 = roi.pos().x(), roi.pos().y()
            for _, point in roi.getLocalHandlePositions():
                xr, yr = point.x(), point.y()
                r.append(int(yr+y0))
                c.append(int(xr+x0))
            if not r or not c:
                return np.array([], dtype=int), np.array([], dtype=int)

            if len(r) == 2:
                rr, cc, val = skimage.draw.line_aa(r[0], c[0], r[1], c[1])
            else:
                rr, cc = skimage.draw.polygon(r, c, shape=self.currentLab2D.shape)

            Y, X = self.currentLab2D.shape
            rr = rr[(rr>=0) & (rr<Y)]
            cc = cc[(cc>=0) & (cc<X)]
            return rr, cc
        elif isinstance(roi, pg.LineROI):
            (_, point1), (_, point2) = roi.getSceneHandlePositions()
            point1 = self.ax1.vb.mapSceneToView(point1)
            point2 = self.ax1.vb.mapSceneToView(point2)
            x1, y1 = int(point1.x()), int(point1.y())
            x2, y2 = int(point2.x()), int(point2.y())
            rr, cc, val = skimage.draw.line_aa(y1, x1, y2, x2)
            return rr, cc
        else:
            x0, y0 = [int(c) for c in roi.pos()]
            w, h = [int(c) for c in roi.size()]
            rr, cc = np.mgrid[y0:y0+h, x0:x0+w]
            return rr.ravel(), cc.ravel()


    def getRoiCoords(self, roi, return_mask=False, z_slice=None):
        """
        Return coordinates covered by roi.

        return_mask=False: (rr, cc)
        return_mask=True:  always (mask, bbox).
            mask is a cropped boolean array containing the roi's exact shape
            (all True for a rectangular ROI, the real silhouette for a
            polygon/line).
            bbox = (r0, r1, c0, c1)
            Placement: 
                full_mask[r0:r1, c0:c1] = mask (2D)
        """
        rr, cc = self.getRoiPixelCoords(roi)

        if not return_mask:
            if self.isSegm3D:
                return z_slice, rr, cc
            return rr, cc

        if rr.size == 0:
            mask = np.zeros((0, 0), bool)
            r0, r1, c0, c1 = 0, 0, 0, 0
        else:
            r0, r1 = int(rr.min()), int(rr.max()) + 1
            c0, c1 = int(cc.min()), int(cc.max()) + 1
            mask = np.zeros((r1 - r0, c1 - c0), bool)
            mask[rr - r0, cc - c0] = True

        bbox = (r0, r1, c0, c1)
        return mask, bbox

    def delROImovingFinished(self, roi: pg.ROI):
        posData = self.data[self.pos_i]
        curr_delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        try:
            curr_idx = curr_delROIs_info['rois'].index(roi)
        except Exception as err:
            for ax in (self.ax1, self.ax2):
                try:
                    ax.removeDelRoiItem(roi)
                    return
                except Exception as err:
                    pass

        update_key = False
        if posData.frame_i > 0:
            prev_delROIs_info = posData.allData_li[posData.frame_i - 1]['delROIs_info']
            try:
                prev_delROIs_info['rois'].index(roi)
                update_key = True
            except ValueError:
                update_key = False
                pass
                        
        update_point_state = False
        if not self.draggingDelROI:
            # check if actual number of points of the poly line is different (i.e. point was added)
            if isinstance(roi, pg.PolyLineROI):
                state = curr_delROIs_info['state'][curr_idx]
                saved_points_len = len(state['points'])
                current_points_len = len(roi.getState()['points'])
                if current_points_len != saved_points_len:
                    update_point_state = True
            if update_point_state is False:
                return
        # check that its not a cancelled poly line

        if update_point_state is False:
            self.draggingDelROI = False
            self.delROIpreviewDelIDsObjDict = dict()  # clear preview dict
            self.currentLab2D[:] = self.delROIoriginalLab # restore original lab
            self.delROIoriginalLab[:] = 0  # clear original lab copy

        new_key = uuid.uuid4()

        curr_state = roi.getState()
        if update_key:
            new_roi = self._createROIfromState(curr_state, new_key)
            self.setDelRoiState(new_roi, curr_state)
            for i in range(posData.frame_i, posData.SizeT):
                frame_delROIs_info = posData.allData_li[i]['delROIs_info']
                try:
                    frame_idx = frame_delROIs_info['rois'].index(roi)
                except ValueError:
                    continue

                frame_delROIs_info['rois'][frame_idx] = new_roi
                frame_delROIs_info['state'][frame_idx] = curr_state
                self.ax2.removeDelRoiItem(roi)
                self.ax1.removeDelRoiItem(roi)

            roi = new_roi
            curr_delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
            curr_idx = curr_delROIs_info['rois'].index(roi)

        self.applyDelROI(roi)
        curr_delROIs_info['state'][curr_idx] = curr_state
        
        QTimer.singleShot(
            300, partial(self.updateDelROIinFutureFrames, roi)
        )

    def applyDelROI(self, roi, frame_i=None, update_img=True, removing_roi=False):
        posData = self.data[self.pos_i]
        
        changed_frame_i = frame_i is not None and frame_i != posData.frame_i
        if changed_frame_i:
            og_frame_i = posData.frame_i
            self.store_data()
            posData.frame_i = frame_i
            self.get_data()
            
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        idx = delROIs_info['rois'].index(roi)
        
        backed_up_masks = delROIs_info['delMasks'][idx]
        backed_up_masks_coords = delROIs_info['delMasksCoords'][idx]
        curr_deleted_IDs = delROIs_info['delIDsROI'][idx]
        
        lab_2D = self.get_2Dlab(posData.lab)
        # readd eleted IDs
        
        mask_roi, bbox_roi = self.getRoiCoords(roi, return_mask=True)
        cutout = np.where(mask_roi, lab_2D[bbox_roi[0]:bbox_roi[1], bbox_roi[2]:bbox_roi[3]], 0)   # same shape as mask, non-roi pixels set to 0
            
        curr_lab = posData.lab
        if not removing_roi:
            touching_IDs = set(self._acdc_regionprops.find_IDs(cutout))
            
            backed_up_masks_new = []
            backed_up_masks_coords_new = []
            curr_deleted_IDs_new = []
                    
            for ID in touching_IDs:
                obj = posData.rp.get_obj_from_ID(ID)
                img = obj.image
                coords = obj.slice
                
                curr_lab[coords][img] = 0
                
                backed_up_masks_new.append(img)
                backed_up_masks_coords_new.append(coords)
                curr_deleted_IDs_new.append(ID)
        
        for delID, mask, coords in zip(curr_deleted_IDs, backed_up_masks, backed_up_masks_coords):
            # make new mask from binary mask and ensureing that there is no overlap with other IDs
            apply_mask = np.logical_and(mask, curr_lab[coords] == 0)
            curr_lab[coords] = np.where(apply_mask, delID, curr_lab[coords])
            
            # check if ID is still inside cutout and should not be readded
            lab_2D = self.get_2Dlab(posData.lab)
            is_still_touching = not removing_roi and np.any(lab_2D[bbox_roi[0]:bbox_roi[1], bbox_roi[2]:bbox_roi[3]][mask_roi] == delID)
            if is_still_touching:
                backed_up_masks_new.append(mask)
                backed_up_masks_coords_new.append(coords)
                curr_deleted_IDs_new.append(delID)
                curr_lab[coords] = np.where(curr_lab[coords] == delID, 0, curr_lab[coords])
                
        if not removing_roi:
            delROIs_info['delMasks'][idx] = backed_up_masks_new
            delROIs_info['delMasksCoords'][idx] = backed_up_masks_coords_new
            delROIs_info['delIDsROI'][idx] = curr_deleted_IDs_new
            
        self.update_rp()
        
        if update_img:
            self.updateAllImages()
        
        if changed_frame_i:
            self.store_data()
            posData.frame_i = og_frame_i
            self.get_data()

    def applyAllDelROI(self):
        posData = self.data[self.pos_i]
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        for roi in delROIs_info['rois']:
            self.applyDelROI(roi)

    def _createROIfromState(self, state, key):
        if 'points' in state and isinstance(state['points'], list) and len(state['points']) > 1:
            # normal poly line
            self.createDelPolyLineRoi(key=key)
            self.polyLineRoi.points = state['points']
            self.addPointsPolyLineRoi(closed=True)
            if len(state['points']) == 2:
                roi = self.replacePolyLineRoiWithLineRoi(self.polyLineRoi)
            else:
                roi = self.polyLineRoi
        else:
            roi, _ = self.createDelROI(key=key)
            if not self.labelsGrad.showLabelsImgAction.isChecked():
                self.ax1.addDelRoiItem(roi, key)
            else:
                self.ax2.addDelRoiItem(roi, key)
        return roi

    def initLoadedDelROI(self):
        og_pos_i = self.pos_i
        for i, posData in enumerate(self.data):
            self.pos_i = i
            self.updateFramePosLabel()
            self.updatePos()
            curr_frame = posData.frame_i
            ROI_key_frame_lookup = posData.loadROIInfo()
            if ROI_key_frame_lookup is None:
                continue
            for ROI_key, frames in ROI_key_frame_lookup.items():
                first_frame = min(frames)
                delROIs_info_first = posData.allData_li[first_frame]['delROIs_info']
                idx = delROIs_info_first['rois'].index(ROI_key)
                state = delROIs_info_first['state'][idx]
                
                roi = self._createROIfromState(state, ROI_key)
                
                for frame in frames:
                    delROIs_info = posData.allData_li[frame]['delROIs_info']
                    idx = delROIs_info['rois'].index(ROI_key)
                    delROIs_info['rois'][idx] = roi
                    
            curr_delROIs_info = posData.allData_li[curr_frame]['delROIs_info']
            for idx, roi in enumerate(curr_delROIs_info['rois']):
                state = curr_delROIs_info['state'][idx]
                self.setDelRoiState(roi, state)
                
            self.addExistingDelROIs()
            self.removeAlldelROIsCurrentFrame()
        self.pos_i = og_pos_i
        self.updateFramePosLabel()
        self.updatePos()

    def moveDelRoisToLeft(self):
        # Move del ROIs to the left image
        for posData in self.data:
            delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
            for roi in delROIs_info['rois']:
                if not self.ax2.isDelRoiItemPresent(roi):
                    continue

                self.ax1.addDelRoiItem(roi, roi.key)
                self.ax2.removeDelRoiItem(roi)

    def setDelRoiState(self, roi: pg.ROI, state):
        roi.sigRegionChanged.disconnect()
        roi.sigRegionChangeFinished.disconnect()
        roi.sigRegionChangeStarted.disconnect()
        roi.setState(state)
        roi.sigRegionChanged.connect(self.delROImoving)
        roi.sigRegionChangeStarted.connect(self.delROIstartedMoving)
        roi.sigRegionChangeFinished.connect(self.delROImovingFinished)

    def addExistingDelROIs(self):
        posData = self.data[self.pos_i]
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        isAx2hidden = not self.labelsGrad.showLabelsImgAction.isChecked()

        for r, roi in enumerate(delROIs_info['rois']):
            if isinstance(roi, pg.PolyLineROI) or isAx2hidden:
                # PolyLine ROIs are only on ax1
                self.ax1.addDelRoiItem(roi, roi.key)
            else:
                # Rect ROI is on ax2 because ax2 is visible
                self.ax2.addDelRoiItem(roi, roi.key)    
            
            self.setDelRoiState(roi, delROIs_info['state'][r])


class LabelRoiGuiElements:
    """Label ROI tool (segmentation with a model inside a ROI) and the
    freehand region actions (whitelist, merge, clear, assign ID).
    """

    def labelRoiIsCircularRadioButtonToggled(self, checked):
        if checked:
            self.labelRoiCircularRadiusSpinbox.setDisabled(False)
        else:
            self.labelRoiCircularRadiusSpinbox.setDisabled(True)

    def updateLabelRoiCircularSize(self, value):
        self.labelRoiCircItemLeft.setSize(value)
        self.labelRoiCircItemRight.setSize(value)

    def updateLabelRoiCircularCursor(self, x, y, checked):
        if not self.labelRoiButton.isChecked():
            return
        if not self.labelRoiIsCircularRadioButton.isChecked():
            return
        if self.labelRoiRunning:
            return

        size = self.labelRoiCircularRadiusSpinbox.value()
        if not checked:
            xx, yy = [], []
        else:
            xx, yy = [x], [y]
        
        if not xx and len(self.labelRoiCircItemLeft.getData()[0]) == 0:
            return

        self.labelRoiCircItemLeft.setData(xx, yy, size=size)
        self.labelRoiCircItemRight.setData(xx, yy, size=size)

    def getLabelRoiImage(self):
        posData = self.data[self.pos_i]

        if self.labelRoiTrangeCheckbox.isChecked():
            start_frame_i = self.labelRoiStartFrameNoSpinbox.value()-1
            stop_frame_n = self.labelRoiStopFrameNoSpinbox.value()
            tRangeLen = stop_frame_n-start_frame_i
        else:
            tRangeLen = 1
        
        if tRangeLen > 1:
            tRange = (start_frame_i, stop_frame_n)
        else:
            tRange = None
        
        if self.isSegm3D:
            if tRangeLen > 1:
                imgData = posData.img_data
            else:
                # Filtered data not existing
                imgData = posData.img_data[posData.frame_i]

            roi_zdepth = self.labelRoiZdepthSpinbox.value()
            if roi_zdepth == posData.SizeZ:
                z0 = 0
                z1 = posData.SizeZ
            elif roi_zdepth == 1:
                z0 = self.zSliceScrollBar.sliderPosition()
                z1 = z0 + 1
            else:
                if roi_zdepth%2 != 0:
                    roi_zdepth +=1
                half_zdepth = int(roi_zdepth/2)
                zc = self.zSliceScrollBar.sliderPosition() + 1
                z0 = zc-half_zdepth
                z0 = z0 if z0>=0 else 0
                z1 = zc+half_zdepth
                z1 = z1 if z1<posData.SizeZ else posData.SizeZ

            if self.labelRoiIsRectRadioButton.isChecked():
                labelRoiSlice = self.labelRoiItem.slice(
                    zRange=(z0,z1), tRange=tRange
                )
            elif self.labelRoiIsFreeHandRadioButton.isChecked():
                labelRoiSlice = self.freeRoiItem.slice(
                    zRange=(z0,z1), tRange=tRange
                )
            elif self.labelRoiIsCircularRadioButton.isChecked():
                labelRoiSlice = self.labelRoiCircItemLeft.slice(
                    zRange=(z0,z1), tRange=tRange
                )
        else:
            if self.labelRoiIsRectRadioButton.isChecked():
                labelRoiSlice = self.labelRoiItem.slice(tRange=tRange)
            elif self.labelRoiIsFreeHandRadioButton.isChecked():
                labelRoiSlice = self.freeRoiItem.slice(tRange=tRange)
            elif self.labelRoiIsCircularRadioButton.isChecked():
                labelRoiSlice = self.labelRoiCircItemLeft.slice(tRange=tRange)
            if tRangeLen > 1:
                imgData = posData.img_data
            else:
                imgData = self.img1.image

        roiImg = imgData[labelRoiSlice]
        if self.labelRoiIsFreeHandRadioButton.isChecked():
            mask = self.freeRoiItem.mask()
        elif self.labelRoiIsCircularRadioButton.isChecked():
            mask = self.labelRoiCircItemLeft.mask()
        else:
            mask = None
        
        if mask is not None:
            # Copy roiImg otherwise we are replacing minimum inside original image
            roiImg = roiImg.copy()
            # Fill outside of freehand roi with minimum of the ROI image
            if tRangeLen > 1:
                for i in range(tRangeLen):
                    ith_roiImg = roiImg[i]
                    if self.isSegm3D:
                        roiImg[i, :, ~mask] = ith_roiImg.min()
                    else:
                        roiImg[i, ~mask] = ith_roiImg.min()
            else:
                if self.isSegm3D:
                    roiImg[:, ~mask] = roiImg.min()
                else:
                    roiImg[~mask] = roiImg.min()

        return roiImg, labelRoiSlice

    def labelRoiTrangeCheckboxToggled(self, checked):
        disabled = not checked
        self.labelRoiStartFrameNoSpinbox.setDisabled(disabled)
        self.labelRoiStopFrameNoSpinbox.setDisabled(disabled)
        self.labelRoiStartFrameNoSpinbox.label.setDisabled(disabled)
        self.labelRoiStopFrameNoSpinbox.label.setDisabled(disabled)
        self.labelRoiToEndFramesAction.setDisabled(disabled)
        self.labelRoiFromCurrentFrameAction.setDisabled(disabled)

        if disabled:
            return

        posData = self.data[self.pos_i]

        self.labelRoiStartFrameNoSpinbox.setValue(posData.frame_i+1)
        self.labelRoiStopFrameNoSpinbox.setValue(posData.SizeT)

    def clearFreehandRoi_cb(self, checked):
        posData = self.data[self.pos_i]
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.clearFreehandRoiButton)
            self.connectLeftClickButtons()

        self.clearFreehandRoiToolbar.setVisible(checked)
        
        if not self.isSegm3D:
            self.clearFreehandRoiToolbar.setZslicesControlEnabled(False)
            return
        
        if not checked:
            return
        
        self.clearFreehandRoiToolbar.setZslicesControlEnabled(
            True, SizeZ=posData.SizeZ
        )

    def labelRoi_cb(self, checked):
        posData = self.data[self.pos_i]
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.labelRoiButton)
            self.connectLeftClickButtons()

            self.labelRoiStartFrameNoSpinbox.setMaximum(posData.SizeT)
            self.labelRoiStopFrameNoSpinbox.setMaximum(posData.SizeT)

            if self.labelRoiActiveWorkers:
                lastActiveWorker = self.labelRoiActiveWorkers[-1]
                self.labelRoiGarbageWorkers.append(lastActiveWorker)
                lastActiveWorker.finished.emit()
                self.logger.info('Collected garbage w5orker (magic labeller).')
            
            self.labelRoiToolbar.setVisible(True)
            if self.isSegm3D:
                self.labelRoiZdepthSpinbox.setDisabled(False)
            else:
                self.labelRoiZdepthSpinbox.setDisabled(True)

            # Start thread and pause it
            self.labelRoiThread = QThread()
            self.labelRoiMutex = QMutex()
            self.labelRoiWaitCond = QWaitCondition()

            labelRoiWorker = gui_workers_segmentation_tracking.LabelRoiWorker(self)

            labelRoiWorker.moveToThread(self.labelRoiThread)
            labelRoiWorker.finished.connect(self.labelRoiThread.quit)
            labelRoiWorker.finished.connect(labelRoiWorker.deleteLater)
            self.labelRoiThread.finished.connect(
                self.labelRoiThread.deleteLater
            )

            labelRoiWorker.finished.connect(self.labelRoiWorkerFinished)
            labelRoiWorker.sigLabellingDone.connect(self.labelRoiDone)
            labelRoiWorker.sigProgressBar.connect(self.workerUpdateProgressbar)

            labelRoiWorker.progress.connect(self.workerProgress)
            labelRoiWorker.critical.connect(self.workerCritical)

            self.labelRoiActiveWorkers.append(labelRoiWorker)

            self.labelRoiThread.started.connect(labelRoiWorker.run)
            self.labelRoiThread.start()

            # Add the rectROI to ax1
            self.ax1.addItem(self.labelRoiItem)
        elif self.initLabelRoiModelDialog is not None:
            # User is using other tools while the dialog is still open 
            # --> we allow this because it's useful to be able to use 
            # the ruler or check things --> do nothing
            pass
        else:
            self.labelRoiToolbar.setVisible(False)
            
            for worker in self.labelRoiActiveWorkers:
                worker._stop()
            while self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()
            
            self.labelRoiItem.setPos((0,0))
            self.labelRoiItem.setSize((0,0))
            self.freeRoiItem.clear()
            self.ax1.removeItem(self.labelRoiItem)
            self.updateLabelRoiCircularCursor(None, None, False)

    def whiteListIDsFreehandRegion(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        alt = modifiers == Qt.AltModifier
        if alt:
            self.freeRoiItem.clear()
            return
        
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            self.freeRoiItem.clear()
            return
        
        self.freeRoiItem.closeCurve()
        self.logger.info(
            'Adding IDs from freehand region to whitelist...')
        
        posData = self.data[self.pos_i]
        zRange = None
        if self.isSegm3D:
            zRange = (0, posData.SizeZ)
        
        onlyEnclosed = event.button() == Qt.MouseButton.LeftButton
        sourceIDs = self.getIDsFreeRoiItem(
            onlyEnclosed=onlyEnclosed, 
            zRange=zRange
        )
        
        current_whitelist = (
            self.whitelistIDsToolbar.whitelistLineEdit.values()
        )
        for ID in sourceIDs:
            current_whitelist.add(ID)
            self.highlightLabelID(ID)
        
        self.whitelistIDsToolbar.whitelistLineEdit.setText(
            current_whitelist
        )

        if posData.whitelist is not None:
            posData.whitelist[posData.frame_i] = current_whitelist

        self.whitelistUpdateTempLayer()
        self.freeRoiItem.clear()


    def mergeObjsFreehandRegion(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        alt = modifiers == Qt.AltModifier
        if alt:
            self.freeRoiItem.clear()
            return
        
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            self.freeRoiItem.clear()
            return
        
        right_click = event.button() == Qt.MouseButton.RightButton
        
        self.freeRoiItem.closeCurve()
        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)
        targetID = None
        if right_click:
            targetID = self.getClickedID(
                xdata, ydata, 
                text='for the merged object.',
                action='released the mouse button'
            )
            if targetID is None:
                self.freeRoiItem.clear()
                return
        
        self.logger.info('Merging objects inside freehand region...')
        
        onlyEnclosed = event.button() == Qt.MouseButton.LeftButton
        sourceIDs = self.applyTargetIDtoFreehandRegion(
            targetID=targetID, 
            onlyEnclosed=onlyEnclosed,
            logger_func=self.mergeIDsToolbar.log
        )
    
        self.mergeIDsToolbar.setIDs(sourceIDs)

    def getIDsFreeRoiItem(self, onlyEnclosed=False, zRange=None):
        posData = self.data[self.pos_i]
        regionSlice = self.freeRoiItem.slice(zRange=zRange)
        mask = self.freeRoiItem.mask()
        regionLab = posData.lab[(...,) + regionSlice].copy()
        if regionLab.ndim == 3:
            mask3d = np.zeros(regionLab.shape, dtype=bool)
            mask3d[:] = mask
            mask = mask3d
            
        if onlyEnclosed:
            regionLab = transformation.clear_objects_not_in_mask(
                regionLab, mask
            )
        else:
            regionLab[..., ~mask] = 0
            
        regionRp = self._acdcRegionProps(
            regionLab,
            precache_centroids=False
        )
        sourceIDs = [obj.label for obj in regionRp]
        
        return sourceIDs

    def applyTargetIDtoFreehandRegion(
            self, 
            targetID=None, 
            onlyEnclosed=False,
            logger_func=None,
            updateImages=True
        ):
        if logger_func is None:
            logger_func = self.logger.info
        
        posData = self.data[self.pos_i]
        zRange = None
        if self.isSegm3D:
            zRange = (0, posData.SizeZ)
        
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(
            False, 
            storeImage=False, 
            storeOnlyZoom=True, 
            zRange=zRange
        )
        
        zProjHow = self.zProjComboBox.currentText()
        isZslice = zProjHow == 'single z-slice'
        only_current_zslice = (
            self.isSegm3D 
            and self.mergeIDsToolbar.isOnlyCurrentZslice()
            and isZslice
        )
        if only_current_zslice:
            z_slice = self.z_lab()
            zRange = (z_slice, z_slice+1)
        
        sourceIDs = self.getIDsFreeRoiItem(
            onlyEnclosed=onlyEnclosed, 
            zRange=zRange
        )

        if not sourceIDs:
            if onlyEnclosed:
                logger_func(
                    'None of the objects in the freehand region are '
                    'fully enclosed'
                )
            else:
                logger_func(
                    'None of the objects are touching the freehand region'
                )
            self.freeRoiItem.clear()
            return
        
        if targetID is None:
            targetID = min(sourceIDs)
        
        if targetID == 0:
            self.deleteIDmiddleClick(sourceIDs, False, False)
            self.update_cca_df_deletedIDs(posData, sourceIDs)
        else:
            self.mergeSelectedIDs(sourceIDs, target_id=targetID)
            
        self.freeRoiItem.clear()
            
        if updateImages:
            self.updateAllImages()
        
        return sourceIDs

    def clearObjsFreehandRegion(self):
        self.logger.info('Clearing objects inside freehand region...')
        
        posData = self.data[self.pos_i]
        zRange = None
        if self.isSegm3D:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if isZslice:
                z_slice = self.z_lab()
                zRange = self.clearFreehandRoiToolbar.zRange(
                    z_slice, posData.SizeZ
                )
            else:
                zRange = (0, posData.SizeZ)
        
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(
            False, 
            storeImage=False, 
            storeOnlyZoom=True, 
            zRange=zRange
        )
        
        clearBorders = (
            self.clearFreehandRoiToolbar
            .clearOnlyEnclosedObjsRadioButton.isChecked()
        )
        
        self.applyTargetIDtoFreehandRegion(
            targetID=0, 
            onlyEnclosed=clearBorders,
            logger_func=self.clearFreehandRoiToolbar.log
        )

    def labelRoiWorkerFinished(self):
        self.logger.info('Magic labeller closed.')
        worker = self.labelRoiActiveWorkers.pop(-1)

    def indexRoiLab(self, roiLab, roiLabSlice, lab, brushID):
        # Delete only objects touching borders in X and Y not in Z
        if self.labelRoiAutoClearBorderCheckbox.isChecked():
            mask = np.zeros(roiLab.shape, dtype=bool)
            mask[..., 1:-1, 1:-1] = True
            roiLab = skimage.segmentation.clear_border(roiLab, mask=mask)

        roiLabMask = roiLab>0
        roiLab[roiLabMask] += (brushID-1)
        if self.labelRoiReplaceExistingObjectsCheckbox.isChecked():
            IDs_touched_by_new_objects = np.unique(lab[roiLabSlice][roiLabMask])
            for ID in IDs_touched_by_new_objects:
                lab[lab==ID] = 0
        
        lab[roiLabSlice][roiLabMask] = roiLab[roiLabMask]
        return lab

    @exception_handler
    def labelRoiDone(self, roiSegmData, isTimeLapse):
        self.setDisabled(False)
        
        posData = self.data[self.pos_i]
        self.setBrushID()

        if isTimeLapse:
            self.progressWin.mainPbar.setMaximum(0)
            self.progressWin.mainPbar.setValue(0)
            current_frame_i = posData.frame_i
            start_frame_i = self.labelRoiStartFrameNoSpinbox.value() - 1
            for i, roiLab in enumerate(roiSegmData):
                frame_i = start_frame_i + i
                lab = posData.allData_li[frame_i]['labels']
                store = True
                if lab is None:
                    if frame_i >= len(posData.segm_data):
                        lab = np.zeros_like(posData.segm_data[0])
                        posData.segm_data = np.append(
                            posData.segm_data, lab[np.newaxis], axis=0
                        )
                    else:
                        lab = posData.segm_data[frame_i]
                    store = False
                roiLabSlice = self.labelRoiSlice[1:]
                lab = self.indexRoiLab(
                    roiLab, roiLabSlice, lab, posData.brushID
                )
                if store:
                    posData.frame_i = frame_i
                    posData.allData_li[frame_i]['labels'] = lab.copy()
                    self.get_data()
                    self.update_rp()
                    self.store_data(autosave=False)
            
            # Back to current frame
            posData.frame_i = current_frame_i
            self.get_data()
        else:
            roiLab = roiSegmData
            posData.lab = self.indexRoiLab(
                roiLab, self.labelRoiSlice, posData.lab, posData.brushID
            )

        self.update_rp() # get roi and set as bbox
        
        # Repeat tracking
        if self.autoIDcheckbox.isChecked():
            self.tracking(enforce=True, assign_unique_new_IDs=False)
        
        self.store_data()
        self.updateAllImages()
        
        self.labelRoiItem.setPos((0,0))
        self.labelRoiItem.setSize((0,0))
        self.freeRoiItem.clear()
        self.logger.info('Magic labeller done!')
        self.app.restoreOverrideCursor()  

        self.labelRoiRunning = False    
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None  
        
        uncheckLabelRoiTRange = (
            self.labelRoiTrangeCheckbox.isChecked()
            and not self.labelRoiTrangeCheckbox.findChild(QAction).isChecked()
        )
        if uncheckLabelRoiTRange:
            self.labelRoiTrangeCheckbox.setChecked(False)

    def storeLabelRoiParams(self, value=None, checked=True):
        checkedRoiType = self.labelRoiTypesGroup.checkedButton().text()
        circRoiRadius = self.labelRoiCircularRadiusSpinbox.value()
        roiZdepth = self.labelRoiZdepthSpinbox.value()
        autoClearBorder = self.labelRoiAutoClearBorderCheckbox.isChecked()
        clearBorder = 'Yes' if autoClearBorder else 'No'
        self.df_settings.at['labelRoi_checkedRoiType', 'value'] = checkedRoiType
        self.df_settings.at['labelRoi_circRoiRadius', 'value'] = circRoiRadius
        self.df_settings.at['labelRoi_roiZdepth', 'value'] = roiZdepth
        self.df_settings.at['labelRoi_autoClearBorder', 'value'] = clearBorder
        self.df_settings.at['labelRoi_replaceExistingObjects', 'value'] = (
            'Yes' if self.labelRoiReplaceExistingObjectsCheckbox.isChecked() 
            else 'No'
        )
        self.df_settings.to_csv(self.settings_csv_path)

    def loadLabelRoiLastParams(self):
        idx = 'labelRoi_checkedRoiType'
        if idx in self.df_settings.index:
            checkedRoiType = self.df_settings.at[idx, 'value']
            for button in self.labelRoiTypesGroup.buttons():
                if button.text() == checkedRoiType:
                    button.setChecked(True)
                    break
        
        idx = 'labelRoi_circRoiRadius'
        if idx in self.df_settings.index:
            circRoiRadius = self.df_settings.at[idx, 'value']
            self.labelRoiCircularRadiusSpinbox.setValue(int(circRoiRadius))
        
        idx = 'labelRoi_roiZdepth'
        if idx in self.df_settings.index:
            roiZdepth = self.df_settings.at[idx, 'value']
            self.labelRoiZdepthSpinbox.setValue(int(roiZdepth))
        
        idx = 'labelRoi_autoClearBorder'
        if idx in self.df_settings.index:
            clearBorder = self.df_settings.at[idx, 'value']
            checked = clearBorder == 'Yes'
            self.labelRoiAutoClearBorderCheckbox.setChecked(checked)
        
        idx = 'labelRoi_replaceExistingObjects'
        if idx in self.df_settings.index:
            val = self.df_settings.at[idx, 'value']
            checked = val == 'Yes'
            self.labelRoiReplaceExistingObjectsCheckbox.setChecked(checked)
        
        if self.labelRoiIsCircularRadioButton.isChecked():
            self.labelRoiCircularRadiusSpinbox.setDisabled(False)

    def showLabelRoiContextMenu(self, event):
        menu = QMenu(self.labelRoiButton)
        action = QAction('Re-initialize magic labeller model...')
        action.triggered.connect(self.initLabelRoiModel)
        menu.addAction(action)
        menu.exec_(QCursor.pos())

    def initLabelRoiModel(self):
        self.app.restoreOverrideCursor()
        # Ask which model
        self.initLabelRoiModelDialog = dialogs_model_selection.QDialogSelectModel(parent=self)
        self.initLabelRoiModelDialog.exec_()
        if self.initLabelRoiModelDialog.cancel:
            self.logger.info('Magic labeller aborted.')
            self.initLabelRoiModelDialog = None
            return True
        self.app.setOverrideCursor(Qt.WaitCursor)
        model_name = self.initLabelRoiModelDialog.selectedModel
        previous_model = self.labelRoiModel
        model = self.repeatSegm(
            model_name=model_name, askSegmParams=True,
            is_label_roi=True
        )
        if model is None:
            self.initLabelRoiModelDialog = None
            return True
        self.labelRoiModel = model
        if previous_model is model:
            # repeatSegm acquired another reference to the existing slot.
            execution = self.modelExecutions.get(model.model_name)
            if execution is not None:
                execution.release(model)
        elif not any(previous_model is stored for stored in self.models):
            self.releaseSegmModel(
                getattr(previous_model, 'model_name', None), previous_model
            )
        self.labelRoiViewCurrentModelAction.setDisabled(False)
        self.initLabelRoiModelDialog = None
        return False

    def labelRoiCancelled(self):
        self.labelRoiRunning = False
        self.app.restoreOverrideCursor() 
        self.labelRoiItem.setPos((0,0))
        self.labelRoiItem.setSize((0,0))
        self.freeRoiItem.clear()
        self.logger.info('Magic labeller process cancelled.')

    def labelRoiCheckStartStopFrame(self):
        if not self.labelRoiTrangeCheckbox.isChecked():
            return True
        
        start_n = self.labelRoiStartFrameNoSpinbox.value()
        stop_n = self.labelRoiStopFrameNoSpinbox.value()
        if start_n <= stop_n:
            return True
        
        blinker = qutils.QControlBlink(
            self.labelRoiStopFrameNoSpinbox, 
            qparent=self
        )
        blinker.start()
        msg = widgets_message_box.myMessageBox()
        txt = html_utils.paragraph("""
            Stop frame number is less than start frame number!<br><br>
            What do you want to do?
        """)
        msg.warning(
            self, 'Stop frame number lower than start', txt, 
            buttonsTexts=('Cancel', 'Segment only current frame')
        )
        if msg.cancel:
            return False
        
        posData = self.data[self.pos_i]
        self.labelRoiStartFrameNoSpinbox.setValue(posData.frame_i+1)
        self.labelRoiStopFrameNoSpinbox.setValue(posData.frame_i+1)

    def labelRoiToEndFramesTriggered(self):
        posData = self.data[self.pos_i]
        self.labelRoiStopFrameNoSpinbox.setValue(posData.SizeT)

    def labelRoiFromCurrentFrameTriggered(self):
        posData = self.data[self.pos_i]
        self.labelRoiStartFrameNoSpinbox.setValue(posData.frame_i+1)

    def labelRoiViewCurrentModel(self):
        from cellacdc import config
        ini_path = os.path.join(
            settings_folderpath, 'last_params_segm_models.ini'
        )
        configPars = config.ConfigParser()
        configPars.read(ini_path)
        model_name = self.labelRoiModel.model_name
        txt = f'Model: <b>{model_name}</b>'
        SECTION = f'{model_name}.init'
        txt = f'{txt}<br><br>[Initialization parameters]<br>'
        for option in configPars.options(SECTION):
            value = configPars[SECTION][option]
            param_txt = f'<i>{option}</i> = {value}<br>'
            txt = f'{txt}{param_txt}'
        
        SECTION = f'{model_name}.segment'
        txt = f'{txt}<br>[Segmentation parameters]<br>'
        for option in configPars.options(SECTION):
            value = configPars[SECTION][option]
            param_txt = f'<i>{option}</i> = {value}<br>'
            txt = f'{txt}{param_txt}'
        
        win = gui_dialogs_segmentation.ViewTextDialog(txt, parent=self)
        win.exec_()
